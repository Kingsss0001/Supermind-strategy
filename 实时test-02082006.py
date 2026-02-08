from datetime import timedelta

# ==================== 1. 策略配置类 ====================
class StrategyConfig:
    """集中管理所有可调参数"""
    
    # 选股参数
    LOOKBACK_DAYS = 233          # 新高回溯天数
    MA_PERIOD = 10               # 回撤均线周期
    
    # 触发条件参数（占位，后续填充具体值）
    PRE_TO_READY_PARAM = {}      # 预备仓→准备仓触发参数
    READY_TO_BUY_PARAM = {}      # 准备仓→买入仓触发参数
    READY_TO_EXIT_PARAM = {}     # 准备仓→出仓触发参数
    BUY_TO_SELL_PARAM = {}       # 买入仓→卖出触发参数
    
    # 资金管理参数（实盘用）
    POSITION_SIZE = 0.1          # 单只股票仓位比例
    MAX_POSITIONS = 10           # 最大持仓数
    
    @classmethod
    def get_config(cls, key):
        """获取配置参数"""
        return getattr(cls, key, None)


# ==================== 2. 数据获取类 ====================
class DataModule:
    """数据获取和计算相关方法"""
    
    @staticmethod
    def get_today_minute_data(stock, start_time, end_time):
        """
        获取当日分钟级数据
        返回: DataFrame 或 None
        """
        try:
            data = get_price(
                [stock], start_time, end_time, '1m',
                ['open', 'high', 'low', 'close', 'volume'],
                False, 'pre', is_panel=0
            )
            if data is not None and stock in data:
                return data[stock]
            return None
        except Exception as e:
            log.error('{} 获取分钟数据失败: {}'.format(stock, str(e)))
            return None
    
    @staticmethod
    def get_daily_data(stock, end_date, bar_count):
        """
        获取日线数据
        返回: DataFrame 或 None
        """
        try:
            data = get_price(
                [stock], None, end_date, '1d',
                ['open', 'high', 'low', 'close', 'volume'],
                False, 'pre', bar_count
            )
            if data is not None and stock in data:
                return data[stock]
            return None
        except Exception as e:
            log.error('{} 获取日线数据失败: {}'.format(stock, str(e)))
            return None
    
    @staticmethod
    def calculate_ma(stock, period, end_date, current_price=None):
        """
        计算均线（支持实时均线）
        修复：使用成交量过滤真实交易日
        
        参数:
            stock: 股票代码
            period: 均线周期（如10日均线传10）
            end_date: 结束日期
            current_price: 如果提供，计算包含当前价的实时均线
        
        返回: float 或 None
        """
        try:
            if current_price is None:
                # ========== 盘后均线计算 ==========
                data = get_price(
                    [stock], None, end_date, '1d',
                    ['close', 'volume'],  # 同时获取成交量
                    False, 'pre', period * 3
                )
                if data is None or stock not in data:
                    return None
                
                df = data[stock]
                
                # 过滤掉成交量为0的日期（休市日）
                valid_days = df[df['volume'] > 100]
                
                if len(valid_days) < period:
                    return None
                
                # 只取最近period个真实交易日
                recent_closes = valid_days['close'].tail(period)
                return recent_closes.mean()
                
            else:
                # ========== 盘中实时均线计算 ==========
                yesterday = get_previous_trading_date(end_date)
                data = get_price(
                    [stock], None, yesterday, '1d',
                    ['close', 'volume'],  # 同时获取成交量
                    False, 'pre', period * 3
                )
                
                if data is None or stock not in data:
                    return None
                
                df = data[stock]
                
                # 过滤掉成交量为0的日期（休市日）
                valid_days = df[df['volume'] > 100]
                
                if len(valid_days) < period - 1:
                    return None
                
                # 只取最近period-1个真实交易日
                recent_prev_close = valid_days['close'].tail(period - 1)
                
                # 实时均线 = (过去period-1天收盘价之和 + 当前价) / period
                ma_value = (recent_prev_close.sum() + current_price) / period
                return ma_value
                
        except Exception as e:
            log.error('{} 计算MA{}失败: {}'.format(stock, period, str(e)))
            return None

    @staticmethod
    def calculate_realtime_dif(stock, current_time, current_price):
        """
        计算盘中实时DIF值
        ⚠️ 关键：将当前价格作为今日收盘价加入EMA计算
        
        参数:
            stock: 股票代码
            current_time: 当前时间
            current_price: 当前价格
        
        返回: float 或 None
        """
        try:
            # 获取历史数据（不含今日）
            data = get_price(
                [stock], None, current_time, '1d',
                ['close', 'volume'],
                False, 'pre', 100
            )
            
            if data is None or stock not in data:
                return None
            
            df = data[stock]
            
            # 过滤真实交易日
            valid_days = df[df['volume'] > 100]
            
            # 取到倒数第二天（昨天为止的历史数据）
            # 因为当日日线数据可能不完整或不存在
            if len(valid_days) >= 2:
                yesterday_closes = valid_days['close'].iloc[:-1]
            else:
                yesterday_closes = valid_days['close']
            
            if len(yesterday_closes) < 26:
                return None
            
            # ⚠️ 关键：将当前价格追加到历史序列
            import pandas as pd
            current_closes = pd.concat([yesterday_closes, pd.Series([current_price])])
            current_closes = current_closes.reset_index(drop=True)
            
            # 用完整序列计算EMA
            ema12 = current_closes.ewm(span=12, adjust=False).mean().iloc[-1]
            ema26 = current_closes.ewm(span=26, adjust=False).mean().iloc[-1]
            
            dif = ema12 - ema26
            
            return dif
            
        except Exception as e:
            log.error('{} 计算实时DIF失败: {}'.format(stock, str(e)))
            return None

    @staticmethod
    def get_highest_price(stock, lookback_days, end_date):
        """
        获取指定周期内的最高价
        修复：使用成交量过滤真实交易日
        
        返回: float 或 None
        """
        try:
            data = get_price(
                [stock], None, end_date, '1d',
                ['high', 'volume'],  # 同时获取成交量
                False, 'pre', lookback_days * 2
            )
            
            if data is None or stock not in data:
                return None
            
            df = data[stock]
            
            # 过滤掉成交量为0的日期（休市日）
            valid_days = df[df['volume'] > 100]
            
            # 取最近lookback_days个真实交易日
            recent_data = valid_days.tail(lookback_days)
            
            if len(recent_data) < lookback_days:
                # 数据不足时至少返回现有数据的最高价
                return recent_data['high'].max() if len(recent_data) > 0 else None
            
            return recent_data['high'].max()
            
        except Exception as e:
            log.error('{} 获取最高价失败: {}'.format(stock, str(e)))
            return None

    @staticmethod
    def calculate_macd_dif(stock, end_date):
        """
        计算MACD的DIF值（12日EMA - 26日EMA）
        修复：使用成交量过滤真实交易日
        
        返回: float 或 None
        """
        try:
            data = get_price(
                [stock], None, end_date, '1d',
                ['close', 'volume'],  # 同时获取成交量
                False, 'pre', 80
            )
            
            if data is None or stock not in data:
                return None
            
            df = data[stock]
            
            # 过滤掉成交量为0的日期（休市日）
            valid_days = df[df['volume'] > 100]
            
            close_prices = valid_days['close']
            
            if len(close_prices) < 26:
                return None
            
            # 计算12日EMA
            ema12 = close_prices.ewm(span=12, adjust=False).mean().iloc[-1]
            
            # 计算26日EMA
            ema26 = close_prices.ewm(span=26, adjust=False).mean().iloc[-1]
            
            # DIF = EMA12 - EMA26
            dif = ema12 - ema26
            
            return dif
            
        except Exception as e:
            log.error('{} 计算DIF失败: {}'.format(stock, str(e)))
            return None

    @staticmethod
    def get_interval_high_low(stock, start_dt, end_dt):
        """
        获取区间内最高价/最低价（分钟级）
        start_dt/end_dt: datetime
        返回: (H, L) 或 (None, None)
        """
        try:
            # 直接取当天分钟数据，再按时间切片
            market_open = start_dt.strftime('%Y-%m-%d') + ' 09:30:00'
            df = DataModule.get_today_minute_data(stock, market_open, end_dt)
            if df is None or len(df) == 0:
                return None, None

            interval = df[(df.index >= start_dt) & (df.index <= end_dt)]
            if interval is None or len(interval) == 0:
                return None, None

            return float(interval['high'].max()), float(interval['low'].min())

        except Exception as e:
            log.error('{} 获取区间高低价失败: {}'.format(stock, str(e)))
            return None, None

    @staticmethod
    def calculate_macd_hist(stock, end_date):
        """
        计算盘后 MACD柱状值（T-1-MACD等）
        口径：MACD_hist = (DIF - DEA) * 2
        """
        try:
            data = get_price([stock], None, end_date, '1d',
                             ['close', 'volume'], False, 'pre', 200)
            if data is None or stock not in data:
                return None

            df = data[stock]
            df = df[df['volume'] > 100]
            if df is None or len(df) < 60:
                return None

            closes = df['close'].values

            ema12 = None
            ema26 = None
            dea = 0.0
            alpha12 = 2.0 / (12 + 1)
            alpha26 = 2.0 / (26 + 1)
            alpha9 = 2.0 / (9 + 1)

            dif_last = 0.0
            for c in closes:
                if ema12 is None:
                    ema12 = c
                    ema26 = c
                else:
                    ema12 = ema12 * (1 - alpha12) + c * alpha12
                    ema26 = ema26 * (1 - alpha26) + c * alpha26

                dif_last = ema12 - ema26
                dea = dea * (1 - alpha9) + dif_last * alpha9

            macd_hist = (dif_last - dea) * 2.0
            return float(macd_hist)

        except Exception as e:
            log.error('{} 计算MACD柱失败: {}'.format(stock, str(e)))
            return None

    @staticmethod
    def calculate_realtime_macd_hist(stock, now_dt, current_price):
        """
        计算盘中实时 MACD柱状值（R-MACD）
        做法：取昨日及更早收盘序列 + 用 current_price 当作“今日收盘”追加
        """
        try:
            yesterday = get_previous_trading_date(now_dt)
            data = get_price([stock], None, yesterday, '1d',
                             ['close', 'volume'], False, 'pre', 200)
            if data is None or stock not in data:
                return None

            df = data[stock]
            df = df[df['volume'] > 100]
            if df is None or len(df) < 60:
                return None

            closes = list(df['close'].values)
            closes.append(float(current_price))

            ema12 = None
            ema26 = None
            dea = 0.0
            alpha12 = 2.0 / (12 + 1)
            alpha26 = 2.0 / (26 + 1)
            alpha9 = 2.0 / (9 + 1)

            dif_last = 0.0
            for c in closes:
                if ema12 is None:
                    ema12 = c
                    ema26 = c
                else:
                    ema12 = ema12 * (1 - alpha12) + c * alpha12
                    ema26 = ema26 * (1 - alpha26) + c * alpha26

                dif_last = ema12 - ema26
                dea = dea * (1 - alpha9) + dif_last * alpha9

            macd_hist = (dif_last - dea) * 2.0
            return float(macd_hist)

        except Exception as e:
            log.error('{} 计算实时MACD柱失败: {}'.format(stock, str(e)))
            return None

    @staticmethod
    def get_highest_price_with_date(stock, lookback_days, end_dt):
        """
        返回：截至 end_dt（含不含由你 get_price 实现决定，但我们用于“截止昨日”）
        向前 lookback_days 内的最高价及其发生日期（YYYY-MM-DD）

        注意：用于 XH-233 的口径是“前233日最高价（不含当天）”，
        所以调用方应传 end_dt = yesterday_dt。
        """
        try:
            end_str = end_dt.strftime('%Y-%m-%d')
            df = DataModule.get_daily_data(stock, end_str, int(lookback_days))
            if df is None or len(df) == 0:
                return None, None

            # 过滤非真实交易日/停牌等异常行（与你MA口径对齐）
            if 'volume' in df.columns:
                df = df[df['volume'] > 100]
            if len(df) == 0:
                return None, None

            # 找最高价及其位置
            max_price = float(df['high'].max())
            idx = df['high'].idxmax()

            # idx 可能是 Timestamp / 字符串 / 数字索引，统一转成日期字符串
            date_str = None
            try:
                # pandas Timestamp / datetime
                if hasattr(idx, 'strftime'):
                    date_str = idx.strftime('%Y-%m-%d')
                else:
                    date_str = str(idx)[:10]
            except Exception:
                date_str = None

            return max_price, date_str

        except Exception as e:
            log.info('【XH-233】{} 获取最高价及日期失败: {}'.format(stock, str(e)))
            return None, None
# ==================== 3. 选股类 ====================
class StockSelector:
    """盘后选股相关方法"""
    
    @staticmethod
    def get_cyb_stocks(date_str):
        """
        获取创业板股票池（剔除ST）
        返回: list
        """
        try:
            all_stocks = get_all_securities('stock', date=date_str)
            cyb_stocks = [
                code for code in all_stocks.index
                if (code.startswith('300') or code.startswith('301'))
                and 'ST' not in all_stocks.loc[code, 'display_name']
            ]
            return cyb_stocks
        except Exception as e:
            log.error('获取创业板股票池失败: {}'.format(str(e)))
            return []
    
    @staticmethod
    def select_new_high_stocks(today_str):
        """
        选出创新高的股票（对齐 02 文档）

        文档条件：
        【数据充足股票（≥233日）】:
        A1: T-0-H >= H-233
        A2: T-0-H > MA-233
        A3: 成交量满足条件（21日均量 > 233日均量*0.99 或 近6日最大量 >= 近55日最大量）
        A4: 创业板非ST（300/301开头且名称不含ST）
        A5: 过去55日出现过 >=10% 的涨幅（历史涨幅）   ← 本次补齐

        【数据不足但符合最低要求（55-232日）】:
        B1: T-0-H > H-ALL
        B2: 同A4
        B3: 同A5（历史涨幅）                           ← 本次补齐
        """
        try:
            # 1. 获取创业板股票池
            cyb_stocks = StockSelector.get_cyb_stocks(today_str)
            if not cyb_stocks:
                log.info('未获取到创业板股票')
                return []

            # 2. 获取历史数据（尽量覆盖233与55窗口）
            start_date = (get_datetime() - timedelta(days=450)).strftime('%Y-%m-%d')

            panel = get_price(
                cyb_stocks,
                start_date,
                today_str,
                '1d',
                ['close', 'high', 'volume'],
                False,
                'pre',
                is_panel=1
            )

            if panel is None:
                log.info('未获取到价格数据')
                return []

            df_close = panel['close']
            df_high = panel['high']
            df_volume = panel['volume']

            # ========== 过滤掉成交量为0/极小的“非真实交易日” ==========
            for stock in df_close.columns:
                valid_mask = df_volume[stock] > 100
                df_close.loc[~valid_mask, stock] = None
                df_high.loc[~valid_mask, stock] = None
                df_volume.loc[~valid_mask, stock] = None

            # 清理全是NaN的行
            df_close = df_close.dropna(how='all')
            df_high = df_high.dropna(how='all')
            df_volume = df_volume.dropna(how='all')

            LOOKBACK = StrategyConfig.LOOKBACK_DAYS  # 233
            MIN_DATA_DAYS = 55

            # 每只股票有效交易日数量
            data_length = df_close.notna().sum()

            # ========== A5/B3：历史涨幅（过去55日出现过>=10%） ==========
            # 口径：以上一交易日收盘价为基准，计算(当日收盘/前收-1)与(当日最高/前收-1)的最大值
            prev_close = df_close.shift(1)
            pct_close = (df_close / prev_close) - 1.0
            pct_high = (df_high / prev_close) - 1.0
            pct_max = pct_high.where(pct_high >= pct_close, pct_close)

            # 最近55个“有效交易日行”（含今天在内；即便含今天也不影响“过去89日出现过”的判定）
            recent_89 = pct_max.tail(89)
            has_10pct_in_89 = (recent_89 >= 0.10).any()   # Series: {stock: bool}
            # ==========================================================

            qualified_stocks = []

            # ========== 数据充足（≥233日）==========
            sufficient_mask = data_length >= LOOKBACK
            sufficient_stocks = data_length[sufficient_mask].index.tolist()

            if sufficient_stocks:
                today_high = df_high.iloc[-1]

                # A1: 今日最高 >= 过去233日最高（不含今日）
                history_high_233 = df_high.tail(LOOKBACK + 1).iloc[:-1].max()
                cond1 = today_high >= history_high_233

                # A2: 今日最高 >= MA233（用close均线近似）
                ma233 = df_close.tail(LOOKBACK).mean()
                cond2 = today_high >= ma233

                # A3: 成交量条件
                vol_ma21 = df_volume.tail(21).mean()
                vol_ma233 = df_volume.tail(233).mean()
                cond3a = vol_ma21 >= vol_ma233 * 0.99

                recent_6_vol_max = df_volume.tail(6).max()
                recent_55_vol_max = df_volume.tail(55).max()
                cond3b = recent_6_vol_max >= recent_55_vol_max

                # A5: 历史涨幅
                cond5 = has_10pct_in_89

                full_cond = cond1 & cond2 & (cond3a | cond3b) & cond5 & ~today_high.isnull()
                full_cond = full_cond & sufficient_mask

                qualified_stocks.extend(full_cond[full_cond].index.tolist())

            # ========== 数据不足但>=55日（55-232日）==========
            insufficient_mask = (data_length >= MIN_DATA_DAYS) & (data_length < LOOKBACK)
            insufficient_stocks = data_length[insufficient_mask].index.tolist()

            if insufficient_stocks:
                today_high = df_high.iloc[-1]
                history_high_all = df_high.iloc[:-1].max()

                # B1: 今日最高 >= 历史最高（不含今日）
                simple_cond = today_high >= history_high_all

                # B3: 历史涨幅（同A5）
                cond5 = has_10pct_in_89

                simple_cond = simple_cond & cond5 & insufficient_mask & ~today_high.isnull()
                qualified_stocks.extend(simple_cond[simple_cond].index.tolist())

            # 数据太少（<55）自动排除
            too_few_mask = data_length < MIN_DATA_DAYS
            too_few_count = int(too_few_mask.sum())

            log.info('数据统计: 充足(≥233日): {} 只 | 不足(55-232日): {} 只 | 太少(<55日): {} 只'.format(
                len(sufficient_stocks),
                len(insufficient_stocks),
                too_few_count
            ))
            log.info('历史涨幅过滤(A5/B3): 满足过去89日>=10%涨幅的股票数: {}'.format(int(has_10pct_in_89.sum())))

            if qualified_stocks:
                log.info('筛选结果: 数据充足组入选 {} 只 | 数据不足组入选 {} 只'.format(
                    len([s for s in qualified_stocks if s in sufficient_stocks]),
                    len([s for s in qualified_stocks if s in insufficient_stocks])
                ))

            return qualified_stocks

        except Exception as e:
            log.error('选股失败: {}'.format(str(e)))
            return []


# ==================== 4. 卖出引擎类 ====================
class SellRuleEngine(object):
    """
    卖出规则引擎（事件驱动/优先级）
    - 盘中：使用分钟数据/实时价判断触发（intraday）
    - 盘后：使用日线与状态变量落地/更新（eod）
    返回统一信号 dict，交由 PoolManager.apply_action_from_sell_signal 执行
    """

    @staticmethod
    def check_intraday(stock):
        rules = []

        # 文档卖出规则（主逻辑）
        rules.append(SellRuleEngine._rule_intraday_psl_system_pos1)

        for fn in rules:
            sig = fn(stock)
            if sig is not None:
                return sig
        return None
        
    @staticmethod
    def check_eod(stock, today_str):
        rules = []

        if hasattr(g, 'enable_tmp_sell_rules') and g.enable_tmp_sell_rules:
            rules.append(SellRuleEngine._rule_eod_force_exit_by_days_5)

        for fn in rules:
            sig = fn(stock, today_str)
            if sig is not None:
                return sig
        return None

    @staticmethod
    def _compute_psl_pbl(interval_low, interval_high, r_ma10, r_ma5):
        """
        PSL/PBL通用计算规则（按02文档）
        基础：
            PSL = L * 0.996
            PBL = H * 1.004
        智能修正（PSL）：
            IF (R-MA-10 <= L <= R-MA-10*1.01): PSL = R-MA-10 * 0.996
            IF (R-MA-5  <= L <= R-MA-5 *1.01): PSL = R-MA-5  * 0.996
        """
        if interval_low is None or interval_high is None:
            return None, None

        L = float(interval_low)
        H = float(interval_high)
        psl = L * 0.996
        pbl = H * 1.004

        try:
            r_ma10 = float(r_ma10)
            r_ma5 = float(r_ma5)

            if r_ma10 <= L <= r_ma10 * 1.01:
                psl = r_ma10 * 0.996

            if r_ma5 <= L <= r_ma5 * 1.01:
                psl = r_ma5 * 0.996

        except Exception:
            pass

        return float(psl), float(pbl)

    @staticmethod
    def _get_realtime_snapshot(stock, now_dt):
        """
        获取盘中所需的统一快照：
        [R-O, R-H, R-L, R-P, R-MA-10, R-MA-5, R-DIF, R-MACD, T-1-C]
        """
        try:
            today_str = now_dt.strftime('%Y-%m-%d')
            market_open = today_str + ' 09:30:00'
            df = DataModule.get_today_minute_data(stock, market_open, now_dt)
            if df is None or len(df) == 0:
                return None

            r_p = float(df['close'].iloc[-1])
            r_o = float(df['open'].iloc[0])
            r_h = float(df['high'].max())
            r_l = float(df['low'].min())

            r_ma10 = DataModule.calculate_ma(stock, 10, now_dt, r_p)
            r_ma5 = DataModule.calculate_ma(stock, 5, now_dt, r_p)
            r_dif = DataModule.calculate_realtime_dif(stock, now_dt, r_p)
            r_macd = DataModule.calculate_realtime_macd_hist(stock, now_dt, r_p)

            if r_ma10 is None or r_ma5 is None or r_dif is None or r_macd is None:
                return None

            # ========== 补齐：T-1-C（昨收）==========
            t1_c = None
            try:
                # 优先用盘前统一缓存（refresh_t1_cache_for_pools 已写入 g.t1_cache）
                if hasattr(g, 't1_cache') and isinstance(g.t1_cache, dict):
                    rec = g.t1_cache.get(stock, None)
                    if isinstance(rec, dict):
                        t1_c = rec.get('c', None)

                if t1_c is None:
                    # 兜底：临时取昨日日线close
                    ydt = get_previous_trading_date(now_dt)
                    ystr = ydt.strftime('%Y-%m-%d')
                    ydf = DataModule.get_daily_data(stock, ystr, 5)
                    if ydf is not None and len(ydf) > 0:
                        if 'volume' in ydf.columns:
                            ydf = ydf[ydf['volume'] > 100]
                        if len(ydf) > 0:
                            t1_c = float(ydf['close'].iloc[-1])
            except Exception:
                t1_c = None
            # ======================================

            return {
                'r_p': r_p,
                'r_o': r_o,
                'r_h': r_h,
                'r_l': r_l,
                'r_ma10': float(r_ma10),
                'r_ma5': float(r_ma5),
                'r_dif': float(r_dif),
                'r_macd': float(r_macd),
                't1_c': float(t1_c) if t1_c is not None else None,
            }

        except Exception as e:
            log.info('【卖出系统】{} 获取盘中快照失败: {}'.format(stock, str(e)))
            return None


    @staticmethod
    def _update_psl_hourly_pos1(stock, now_dt, snap):
        """
        满仓（POS=1.0）整点行为：刷新 PSL/PSL-T 或清空

        命中策略：First（自上而下命中即停止）
        1) L1-B → 设置PSL, PSL-T='half', PSL=max(旧,新) → RETURN
        2) L1-D → 清空PSL, PSL-T=None → RETURN
        3) L1-C → 设置PSL, PSL-T='full', PSL=max(旧,新) → RETURN
        4) L1-A → 设置PSL, PSL-T='half', PSL=max(旧,新) → RETURN
        5) L2-4B → 设置PSL, PSL-T='full', PSL=max(旧,新) → RETURN
        6) L2-2 OR L2-4A → 设置PSL, PSL-T='half', PSL=max(旧,新) → RETURN
        7) L2-1 OR L2-3 → 清空PSL, PSL-T=None → RETURN
        else → 清空PSL, PSL-T=None → RETURN

        ✅ STEP3-关键修复：
        - 整点评估完成后必须“显式告知调用方已处理”，以便调用方立即 return，杜绝整点成交。
        - 返回值：True=本次为整点且已完成评估（设线/清线之一）；False=非整点或未完成评估
        """
        hm = now_dt.strftime('%H:%M')
        hourly_list = ['10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30']
        if hm not in hourly_list:
            return False

        # 防重复
        if not hasattr(g, 'sell_trg_time'):
            g.sell_trg_time = {}
        if g.sell_trg_time.get(stock, None) == hm:
            return False

        # T+1前置
        days = int(g.buy_days.get(stock, 1)) if hasattr(g, 'buy_days') else 1
        if days < 2:
            return False

        # 容器
        if not hasattr(g, 'sell_psl'):
            g.sell_psl = {}
        if not hasattr(g, 'sell_psl_t'):
            g.sell_psl_t = {}
        if not hasattr(g, 'sell_psl_src'):
            g.sell_psl_src = {}   # {stock: 'L1-A'/'L2-4B'/...} 记录“当前PSL来自哪个条件”

        def _fmt2(x):
            try:
                if x is None:
                    return 'None'
                return '{:.2f}'.format(float(x))
            except Exception:
                return str(x)

        # 取 T-1 指标
        t1_h = g.buy_t1_h.get(stock, None) if hasattr(g, 'buy_t1_h') else None
        t1_dif = g.buy_t1_dif.get(stock, None) if hasattr(g, 'buy_t1_dif') else None
        t1_macd = g.yesterday_macd.get(stock, None) if hasattr(g, 'yesterday_macd') else None
        if t1_h is None or t1_dif is None or t1_macd is None:
            return False
        t1_h = float(t1_h)
        t1_dif = float(t1_dif)
        t1_macd = float(t1_macd)

        x_h = g.buy_xh.get(stock, None) if hasattr(g, 'buy_xh') else None
        xh_233 = g.buy_xh233.get(stock, None) if hasattr(g, 'buy_xh233') else None

        # 区间：整点/半点往前30分钟
        start_dt = now_dt - timedelta(minutes=30)
        interval_h, interval_l = DataModule.get_interval_high_low(stock, start_dt, now_dt)
        psl_new, _ = SellRuleEngine._compute_psl_pbl(interval_l, interval_h, snap['r_ma10'], snap['r_ma5'])
        if psl_new is None:
            return False

        # ✅ 到这里说明“整点评估”确实要执行，先落 sell_trg_time，随后无论设线/清线都算处理完成
        g.sell_trg_time[stock] = hm

        # ===== 计算条件 =====
        l1_a = (snap['r_o'] > t1_h)

        l1_b = False
        gap_ok = False
        if x_h is not None and xh_233 is not None:
            try:
                x_h = float(x_h)
                xh_233 = float(xh_233)
                gap_ok = SellRuleEngine._is_gap_gt1_between_today_and_xh233_peak_date(stock, now_dt)
                l1_b = (snap['r_h'] >= xh_233 * 0.999) and (xh_233 * 0.999 > x_h) and gap_ok
            except Exception:
                l1_b = False
                gap_ok = False

        l1_c = (float(interval_l) < snap['r_ma10']) and (snap['r_dif'] < t1_dif)
        l1_d = (snap['r_l'] < snap['r_ma10']) and (snap['r_dif'] >= t1_dif)

        l2_1 = (snap['r_h'] >= t1_h) and (snap['r_dif'] >= t1_dif)
        l2_2 = (snap['r_h'] >= t1_h) and (snap['r_dif'] < t1_dif)
        l2_3 = (snap['r_h'] < t1_h) and (snap['r_macd'] >= t1_macd)
        l2_4a = (days == 2) and (snap['r_h'] < t1_h) and (snap['r_macd'] < t1_macd)
        l2_4b = (days > 2) and (snap['r_h'] < t1_h) and (snap['r_macd'] < t1_macd)

        old_psl = g.sell_psl.get(stock, None)

        def _set_psl(src_label, psl_t_value):
            new_val = float(psl_new)
            if old_psl is None:
                final_psl = new_val
            else:
                try:
                    final_psl = max(float(old_psl), new_val)
                except Exception:
                    final_psl = new_val

            g.sell_psl[stock] = final_psl
            g.sell_psl_t[stock] = psl_t_value
            g.sell_psl_src[stock] = src_label

            log.info('【整点评估触发】{} | hm:{} | 触发条件:{} | 设置PSL:{:.2f} | PSL-T:{}'.format(
                stock, hm, src_label, float(final_psl), str(psl_t_value)
            ))
            return True

        def _clear_psl(src_label):
            g.sell_psl[stock] = None
            g.sell_psl_t[stock] = None
            g.sell_psl_src[stock] = None
            log.info('【整点评估触发】{} | hm:{} | 触发条件:{} | 清空PSL/PSL-T'.format(
                stock, hm, src_label
            ))
            return True

        # ===== First 优先级执行 + 输出“条件相关数据” =====
        if l1_b:
            log.info('【触发条件】{} | hm:{} | L1-B | R-H:{} | XH-233:{} | X-H:{} | gap_ok:{}'.format(
                stock, hm, _fmt2(snap['r_h']), _fmt2(xh_233), _fmt2(x_h), str(gap_ok)
            ))
            return _set_psl('L1-B', 'half')

        if l1_d:
            log.info('【触发条件】{} | hm:{} | L1-D | R-L:{} | R-MA10:{} | R-DIF:{} | T-1-DIF:{}'.format(
                stock, hm, _fmt2(snap['r_l']), _fmt2(snap['r_ma10']), _fmt2(snap['r_dif']), _fmt2(t1_dif)
            ))
            return _clear_psl('L1-D')

        if l1_c:
            log.info('【触发条件】{} | hm:{} | L1-C | L(30m):{} | R-MA10:{} | R-DIF:{} | T-1-DIF:{}'.format(
                stock, hm, _fmt2(interval_l), _fmt2(snap['r_ma10']), _fmt2(snap['r_dif']), _fmt2(t1_dif)
            ))
            return _set_psl('L1-C', 'full')

        if l1_a:
            log.info('【触发条件】{} | hm:{} | L1-A | R-O:{} | T-1-H:{}'.format(
                stock, hm, _fmt2(snap['r_o']), _fmt2(t1_h)
            ))
            return _set_psl('L1-A', 'half')

        if l2_4b:
            log.info('【触发条件】{} | hm:{} | L2-4B | DAYS:{} | R-H:{} | T-1-H:{} | R-MACD:{} | T-1-MACD:{}'.format(
                stock, hm, str(days), _fmt2(snap['r_h']), _fmt2(t1_h), _fmt2(snap['r_macd']), _fmt2(t1_macd)
            ))
            return _set_psl('L2-4B', 'full')

        if l2_2 or l2_4a:
            if l2_2:
                log.info('【触发条件】{} | hm:{} | L2-2 | R-H:{} | T-1-H:{} | R-DIF:{} | T-1-DIF:{}'.format(
                    stock, hm, _fmt2(snap['r_h']), _fmt2(t1_h), _fmt2(snap['r_dif']), _fmt2(t1_dif)
                ))
                return _set_psl('L2-2', 'half')
            log.info('【触发条件】{} | hm:{} | L2-4A | DAYS:{} | R-H:{} | T-1-H:{} | R-MACD:{} | T-1-MACD:{}'.format(
                stock, hm, str(days), _fmt2(snap['r_h']), _fmt2(t1_h), _fmt2(snap['r_macd']), _fmt2(t1_macd)
            ))
            return _set_psl('L2-4A', 'half')

        if l2_1 or l2_3:
            if l2_1:
                log.info('【触发条件】{} | hm:{} | L2-1 | R-H:{} | T-1-H:{} | R-DIF:{} | T-1-DIF:{}'.format(
                    stock, hm, _fmt2(snap['r_h']), _fmt2(t1_h), _fmt2(snap['r_dif']), _fmt2(t1_dif)
                ))
                return _clear_psl('L2-1')
            log.info('【触发条件】{} | hm:{} | L2-3 | R-H:{} | T-1-H:{} | R-MACD:{} | T-1-MACD:{}'.format(
                stock, hm, _fmt2(snap['r_h']), _fmt2(t1_h), _fmt2(snap['r_macd']), _fmt2(t1_macd)
            ))
            return _clear_psl('L2-3')

        # else：清空
        return _clear_psl('NONE')

    @staticmethod
    def _update_psl_hourly_pos05(stock, now_dt, snap, t1_h, t1_dif, t1_macd):
        """
        半仓（POS=0.5）整点行为：刷新 PSL/PBL 或 清空（按02文档）

        整点列表：
        ['10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30']

        命中策略：First（自上而下命中即停止）
        1) (L0 AND 子条件组) -> 设置PBL, PBL-T='half', PBL=min(旧,新) -> RETURN
        2) L1-D OR L2-1 OR (L0成立但未命中子条件组) -> 清空PSL/PSL-T -> RETURN
        3) L1-C -> 设置PSL, PSL-T='empty', PSL=max(旧,新) -> RETURN
        4) ELSE -> 设置PSL, PSL-T='empty', PSL=max(旧,新) -> RETURN

        说明：
        - L0基础条件：R-MACD >= T-1-MACD
        - 子条件组（三选一）：
          A) gap <= 3个交易日（T-0 - XH233-D <= 3）
          B) gap > 3 且 XH-233 > X-H 且 R-H < XH-233 * 0.97
          C) XH-233 <= X-H
        - PSL 只涨不跌（max）
        - PBL 只跌不涨（min）
        - 返回值：True=本次为整点且已完成评估；False=非整点或未完成评估

        ✅ 20260208补丁：半仓锁例外2闭环
        - 半仓锁当日，整点命中 L1-C：只“授权当日可触线卖剩余半仓”（整点不卖）
        """
        hm = now_dt.strftime('%H:%M')
        today_str = now_dt.strftime('%Y-%m-%d')
        hourly_list = ['10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30']
        if hm not in hourly_list:
            return False

        # 防重复
        if not hasattr(g, 'sell_trg_time'):
            g.sell_trg_time = {}
        if g.sell_trg_time.get(stock, None) == hm:
            return False

        # T+1前置（保险起见）
        days = int(g.buy_days.get(stock, 1)) if hasattr(g, 'buy_days') else 1
        if days < 2:
            return False

        # 容器确保存在
        if not hasattr(g, 'sell_psl'):
            g.sell_psl = {}
        if not hasattr(g, 'sell_psl_t'):
            g.sell_psl_t = {}
        if not hasattr(g, 'sell_psl_src'):
            g.sell_psl_src = {}
        if not hasattr(g, 'sell_pbl'):
            g.sell_pbl = {}
        if not hasattr(g, 'sell_pbl_t'):
            g.sell_pbl_t = {}
        if not hasattr(g, 'sell_pbl_src'):
            g.sell_pbl_src = {}

        # ✅ 半仓锁授权字典
        if not hasattr(g, 'half_lock_l1c_allow_date'):
            g.half_lock_l1c_allow_date = {}

        def _fmt2(x):
            try:
                if x is None:
                    return 'None'
                return '{:.2f}'.format(float(x))
            except Exception:
                return str(x)

        # 取 T-1 指标（必须）
        if t1_h is None or t1_dif is None or t1_macd is None:
            return False
        try:
            t1_h = float(t1_h)
            t1_dif = float(t1_dif)
            t1_macd = float(t1_macd)
        except Exception:
            return False

        # 区间：整点/半点往前30分钟
        start_dt = now_dt - timedelta(minutes=30)
        interval_h, interval_l = DataModule.get_interval_high_low(stock, start_dt, now_dt)
        psl_new, pbl_new = SellRuleEngine._compute_psl_pbl(interval_l, interval_h, snap['r_ma10'], snap['r_ma5'])
        if psl_new is None or pbl_new is None:
            return False

        # ✅ 到这里说明“整点评估”确实要执行，先落 sell_trg_time
        g.sell_trg_time[stock] = hm

        # 半仓锁判断（例外2只在锁内生效）
        half_locked_today = False
        try:
            if hasattr(g, 'pos_half_lock_date') and isinstance(g.pos_half_lock_date, dict):
                half_locked_today = (g.pos_half_lock_date.get(stock, None) == today_str)
        except Exception:
            half_locked_today = False

        # 取锚点值（半仓阶段依赖：XH-233 / XH233-D / X-H）
        xh_233 = g.buy_xh233.get(stock, None) if hasattr(g, 'buy_xh233') else None
        xh_233_date = g.buy_xh233_date.get(stock, None) if hasattr(g, 'buy_xh233_date') else None
        x_h = g.buy_xh.get(stock, None) if hasattr(g, 'buy_xh') else None

        # ===== 条件计算 =====
        # L0基础：R-MACD >= T-1-MACD
        l0_base = False
        try:
            l0_base = float(snap['r_macd']) >= float(t1_macd)
        except Exception:
            l0_base = False

        # gap（交易日差）：today - XH233-D 的交易日步数
        gap_td = None
        if xh_233_date:
            try:
                if xh_233_date == today_str:
                    gap_td = 0
                else:
                    cursor = now_dt
                    gap = 0
                    for _ in range(400):
                        prev_dt = get_previous_trading_date(cursor)
                        gap += 1
                        prev_str = prev_dt.strftime('%Y-%m-%d')
                        if prev_str == xh_233_date:
                            break
                        cursor = prev_dt
                    else:
                        gap = None
                    gap_td = gap
            except Exception:
                gap_td = None

        # 子条件组（需锚点数据）
        sub_a = sub_b = sub_c = False
        if l0_base and (xh_233 is not None) and (x_h is not None) and (gap_td is not None):
            try:
                xh_233_f = float(xh_233)
                x_h_f = float(x_h)
                r_h_f = float(snap['r_h'])

                # A) gap <= 3个交易日
                sub_a = (gap_td is not None and gap_td <= 3)

                # B) gap > 3 且 XH-233 > X-H 且 R-H < XH-233 * 0.97
                sub_b = (gap_td is not None and gap_td > 3 and xh_233_f > x_h_f and r_h_f < xh_233_f * 0.97)

                # C) XH-233 <= X-H
                sub_c = (xh_233_f <= x_h_f)

            except Exception:
                sub_a = sub_b = sub_c = False

        l0_hit_group = (sub_a or sub_b or sub_c)

        # L1-D / L2-1 / L1-C
        l1_d = (float(snap['r_l']) < float(snap['r_ma10'])) and (float(snap['r_dif']) >= float(t1_dif))
        l2_1 = (float(snap['r_h']) >= float(t1_h)) and (float(snap['r_dif']) >= float(t1_dif))
        l1_c = (float(interval_l) < float(snap['r_ma10'])) and (float(snap['r_dif']) < float(t1_dif))

        old_psl = g.sell_psl.get(stock, None)
        old_pbl = g.sell_pbl.get(stock, None)

        def _set_psl(src_label):
            new_val = float(psl_new)
            if old_psl is None:
                final_psl = new_val
            else:
                try:
                    final_psl = max(float(old_psl), new_val)
                except Exception:
                    final_psl = new_val

            g.sell_psl[stock] = final_psl
            g.sell_psl_t[stock] = 'empty'
            g.sell_psl_src[stock] = src_label

            # ✅ 例外2授权：半仓锁当日整点命中 L1-C，仅授予“当日可触线卖剩余半仓”资格（整点不卖）
            if half_locked_today and src_label == 'L1-C':
                g.half_lock_l1c_allow_date[stock] = today_str
                log.info('【半仓锁-例外2授权】{} | 日期:{} | hm:{} | 命中L1-C => 授权当日可触线卖剩余半仓'.format(
                    stock, today_str, hm
                ))

            log.info('【整点评估触发-半仓】{} | hm:{} | 触发条件:{} | 设置PSL:{:.2f} | PSL-T:empty'.format(
                stock, hm, src_label, float(final_psl)
            ))
            return True

        def _clear_psl(src_label):
            g.sell_psl[stock] = None
            g.sell_psl_t[stock] = None
            g.sell_psl_src[stock] = None

            log.info('【整点评估触发-半仓】{} | hm:{} | 触发条件:{} | 清空PSL/PSL-T'.format(
                stock, hm, src_label
            ))
            return True

        def _set_pbl(src_label):
            new_val = float(pbl_new)
            if old_pbl is None:
                final_pbl = new_val
            else:
                try:
                    final_pbl = min(float(old_pbl), new_val)   # PBL只跌不涨
                except Exception:
                    final_pbl = new_val

            g.sell_pbl[stock] = final_pbl
            g.sell_pbl_t[stock] = 'half'
            g.sell_pbl_src[stock] = src_label

            log.info('【整点评估触发-半仓】{} | hm:{} | 触发条件:{} | 设置PBL:{:.2f} | PBL-T:half | oldPBL:{} | gap:{} | XH-233:{} | X-H:{} | R-H:{}'.format(
                stock, hm, src_label, float(final_pbl),
                _fmt2(old_pbl), str(gap_td), _fmt2(xh_233), _fmt2(x_h), _fmt2(snap['r_h'])
            ))
            return True

        # ===== First 优先级执行 =====
        # 1) L0命中子条件组 -> 设置PBL
        if l0_base and l0_hit_group:
            sub_tag = 'A(gap<=3)' if sub_a else ('B(gap>3&XH233>XH&R-H<0.97)' if sub_b else 'C(XH233<=XH)')
            return _set_pbl('L0-' + sub_tag)

        # 2) L1-D OR L2-1 OR (L0成立但未命中子条件组) -> 清空PSL
        if l1_d:
            return _clear_psl('L1-D')
        if l2_1:
            return _clear_psl('L2-1')
        if l0_base and (not l0_hit_group):
            return _clear_psl('L0_BASE_ONLY')

        # 3) L1-C -> 设置PSL(empty)（并在半仓锁当日写授权）
        if l1_c:
            return _set_psl('L1-C')

        # 4) ELSE -> 设置PSL(empty)
        return _set_psl('ELSE')

    @staticmethod
    def _rule_intraday_psl_system_pos1(stock):
        """
        02文档阶段5：卖出系统（覆盖 POS=1.0 与 POS=0.5）

        方案B（更保守）：
        - 每天 14:55 之后（非 14:55 当分钟），禁止所有盘中触线成交（PSL卖出 / PBL回补 / 半仓锁例外2触线卖出）
        - 仅保留 14:55 兜底逻辑可运行

        本次修正：
        - 半仓锁当日（POS从1变0.5的当日）：
          仅允许三种例外：
            1) 14:55 L1-A => 0.5→1.0
            2) 整点 L1-C 授权后，非整点触线 => 0.5→0
            3) 14:55 L1-F => 0.5→0
          其余一律禁止（特别是：锁内 14:55 不允许 L0 回补，也不允许 ELSE 卖出）
        """
        if not hasattr(g, 'buy_pool') or stock not in g.buy_pool:
            return None
        if not hasattr(g, 'buy_pos') or stock not in g.buy_pos:
            return None

        pos = float(g.buy_pos.get(stock, 1.0))
        days = int(g.buy_days.get(stock, 1)) if hasattr(g, 'buy_days') else 1
        if days < 2:
            return None

        now_dt = get_datetime()
        today_str = now_dt.strftime('%Y-%m-%d')
        hm = now_dt.strftime('%H:%M')

        snap = SellRuleEngine._get_realtime_snapshot(stock, now_dt)
        if snap is None:
            return None

        t1_h = g.buy_t1_h.get(stock, None) if hasattr(g, 'buy_t1_h') else None
        t1_dif = g.buy_t1_dif.get(stock, None) if hasattr(g, 'buy_t1_dif') else None
        t1_macd = g.yesterday_macd.get(stock, None) if hasattr(g, 'yesterday_macd') else None
        if t1_h is None or t1_dif is None or t1_macd is None:
            return None
        t1_h = float(t1_h)
        t1_dif = float(t1_dif)
        t1_macd = float(t1_macd)

        rebuy_date = g.rebuy_date.get(stock, None) if hasattr(g, 'rebuy_date') else None
        is_rebuy_day = (rebuy_date == today_str)

        if not hasattr(g, 'pos_half_lock_date'):
            g.pos_half_lock_date = {}
        half_locked_today = (g.pos_half_lock_date.get(stock, None) == today_str)

        if not hasattr(g, 'sell_1455_done'):
            g.sell_1455_done = {}
        is_1455_point = (hm == '14:55')
        already_done = (g.sell_1455_done.get(stock, None) == today_str)

        hourly_list = ['10:00', '10:30', '11:00', '11:30', '13:30', '14:00', '14:30']
        is_hourly_point = (hm in hourly_list)

        if not hasattr(g, 'sell_psl'):
            g.sell_psl = {}
        if not hasattr(g, 'sell_psl_t'):
            g.sell_psl_t = {}
        if not hasattr(g, 'sell_psl_src'):
            g.sell_psl_src = {}
        if not hasattr(g, 'sell_pbl'):
            g.sell_pbl = {}
        if not hasattr(g, 'sell_pbl_t'):
            g.sell_pbl_t = {}
        if not hasattr(g, 'sell_trg_time'):
            g.sell_trg_time = {}

        # 半仓锁例外2授权容器
        if not hasattr(g, 'half_lock_l1c_allow_date'):
            g.half_lock_l1c_allow_date = {}

        # ============================================================
        # ======================= POS = 1.0 ==========================
        # ============================================================
        if pos == 1.0:
            hourly_done = SellRuleEngine._update_psl_hourly_pos1(stock, now_dt, snap)
            if hourly_done is True:
                return None

            # 整点只设线/清线，不成交
            if is_hourly_point:
                return None

            # 14:55兜底（满仓）——已对齐版本
            if is_1455_point and (not already_done):
                g.sell_1455_done[stock] = today_str

                rp_ge_ro = (float(snap['r_p']) >= float(snap['r_o']))
                l1_a = (float(snap['r_o']) > float(t1_h))

                x_h = g.buy_xh.get(stock, None) if hasattr(g, 'buy_xh') else None
                xh_233 = g.buy_xh233.get(stock, None) if hasattr(g, 'buy_xh233') else None
                l1_b = False
                if x_h is not None and xh_233 is not None:
                    try:
                        x_h_f = float(x_h)
                        xh_233_f = float(xh_233)
                        gap_ok = SellRuleEngine._is_gap_gt1_between_today_and_xh233_peak_date(stock, now_dt)
                        l1_b = (float(snap['r_h']) >= xh_233_f * 0.999) and (xh_233_f * 0.999 > x_h_f) and gap_ok
                    except Exception:
                        l1_b = False

                l1_d = (float(snap['r_l']) < float(snap['r_ma10'])) and (float(snap['r_dif']) >= float(t1_dif))
                l2_1 = (float(snap['r_h']) >= float(t1_h)) and (float(snap['r_dif']) >= float(t1_dif))

                l2_4b = (days > 2) and (float(snap['r_h']) < float(t1_h)) and (float(snap['r_macd']) < float(t1_macd))
                l1_f = (float(snap['r_p']) < float(snap['r_ma10']))

                if l1_d or l2_1 or (rp_ge_ro and (l1_a or l1_b)):
                    return None

                if (not is_rebuy_day) and (l2_4b or l1_f):
                    return {'action': 'SELL_ALL', 'destination': 'pre', 'rule': '1455_FULL(L2-4B_or_L1-F)', 'time': hm, 'metrics': None}

                return {'action': 'SELL_HALF', 'destination': None, 'rule': 'ELSE(1455_HALF)', 'time': hm, 'metrics': None}

            # 方案B硬闸：14:55之后（非14:55当分钟）禁止所有盘中触线成交
            if hm >= '14:55':
                return None

            # PSL触线（仅允许14:54及之前）
            psl = g.sell_psl.get(stock, None)
            psl_t = g.sell_psl_t.get(stock, None)
            if psl is not None and psl_t is not None:
                try:
                    psl_f = float(psl)
                except Exception:
                    psl_f = None

                if psl_f is not None and float(snap['r_p']) <= psl_f:
                    src = g.sell_psl_src.get(stock, None) or 'PSL'
                    if is_rebuy_day:
                        return {'action': 'SELL_HALF', 'destination': None, 'rule': src, 'time': hm, 'psl': psl_f, 'r_p': snap['r_p']}
                    if psl_t == 'full':
                        return {'action': 'SELL_ALL', 'destination': 'pre', 'rule': src, 'time': hm, 'psl': psl_f, 'r_p': snap['r_p']}
                    if psl_t == 'half':
                        return {'action': 'SELL_HALF', 'destination': None, 'rule': src, 'time': hm, 'psl': psl_f, 'r_p': snap['r_p']}

            return None

        # ============================================================
        # ======================= POS = 0.5 ==========================
        # ============================================================
        if pos == 0.5:
            # 14:55兜底（半仓）
            if is_1455_point and (not already_done):
                g.sell_1455_done[stock] = today_str

                metrics = {
                    'r_o': snap['r_o'],
                    'r_h': snap['r_h'],
                    'r_l': snap['r_l'],
                    'r_p': snap['r_p'],
                    'r_ma10': snap['r_ma10'],
                    'r_ma5': snap['r_ma5'],
                    'r_dif': snap['r_dif'],
                    'r_macd': snap['r_macd'],
                    't1_h': t1_h,
                    't1_dif': t1_dif,
                    't1_macd': t1_macd
                }

                PoolManager.record_sell_1455_log(
                    stock=stock,
                    now_dt=now_dt,
                    hm=hm,
                    pos=pos,
                    days=days,
                    bp=None,
                    rule='1455_HALF_EVAL',
                    metrics=metrics,
                    extra=None
                )

                l1_a_1455 = (float(snap['r_o']) > float(t1_h))
                l1_f_1455 = (float(snap['r_p']) < float(snap['r_ma10']))

                # 半仓锁当日：仅允许三种例外中的(1)与(3)在14:55发生
                if half_locked_today:
                    # 例外1：14:55 触发 L1-A => 回补半仓（0.5→1.0）
                    if l1_a_1455:
                        return {'action': 'REBUY', 'destination': None, 'rule': 'L1-A(1455_HALF_LOCK)', 'time': hm, 'metrics': metrics}
                    # 例外3：14:55 触发 L1-F => 卖出剩余半仓（0.5→0）
                    if l1_f_1455:
                        return {'action': 'SELL_ALL', 'destination': 'pre', 'rule': 'L1-F(1455_HALF_LOCK)', 'time': hm, 'metrics': metrics}
                    # 其余情况：锁内14:55不允许再触发其它动作（不允许L0回补，也不允许ELSE卖出）
                    return None

                # 非半仓锁当日：保持原半仓14:55兜底逻辑（L1-A 或 L0 => 回补；否则按文档继续判断）
                l0_1455 = (
                    float(snap['r_h']) >= float(t1_h)
                    and float(snap['r_p']) > float(snap['r_o'])
                    and float(snap['r_dif']) >= float(t1_dif)
                    and float(snap['r_macd']) >= float(t1_macd)
                    and ((float(snap['r_p']) - float(snap['t1_c'])) / float(snap['t1_c'])) < 0.19
                ) if snap.get('t1_c', None) not in (None, 0) else False

                if l1_a_1455 or l0_1455:
                    rule_tag = 'L1-A(1455_HALF)' if l1_a_1455 else 'L0(1455_HALF)'
                    return {'action': 'REBUY', 'destination': None, 'rule': rule_tag, 'time': hm, 'metrics': metrics}

                l1_d = (float(snap['r_l']) < float(snap['r_ma10'])) and (float(snap['r_dif']) >= float(t1_dif))
                l2_1 = (float(snap['r_h']) >= float(t1_h)) and (float(snap['r_dif']) >= float(t1_dif))
                if l1_d or l2_1:
                    return None

                return {'action': 'SELL_ALL', 'destination': 'pre', 'rule': 'ELSE(1455_HALF)', 'time': hm, 'metrics': metrics}

            # 方案B硬闸：14:55之后（非14:55当分钟）禁止所有盘中触线成交（含PBL回补、PSL卖出、例外2触线卖出）
            if hm >= '14:55':
                return None

            # 半仓整点：只允许设线/清线（禁止同bar成交）
            hourly_done_half = SellRuleEngine._update_psl_hourly_pos05(stock, now_dt, snap, t1_h, t1_dif, t1_macd)
            if hourly_done_half is True:
                return None
            if is_hourly_point:
                return None

            # 半仓锁当日：默认禁止；仅例外2（L1-C整点授权）允许“后续非整点PSL触线卖出剩余半仓”
            if half_locked_today:
                allow_date = g.half_lock_l1c_allow_date.get(stock, None)

                if allow_date != today_str:
                    return None

                src = g.sell_psl_src.get(stock, None)
                if src != 'L1-C':
                    return None

                psl = g.sell_psl.get(stock, None)
                psl_t = g.sell_psl_t.get(stock, None)
                if psl is not None and psl_t is not None:
                    try:
                        psl_f = float(psl)
                    except Exception:
                        psl_f = None

                    if psl_f is not None and float(snap['r_p']) <= psl_f:
                        return {'action': 'SELL_ALL', 'destination': 'pre', 'rule': 'L1-C(授权触线)', 'time': hm, 'psl': psl_f, 'r_p': snap['r_p']}

                return None

            # 非half_locked_today：正常PBL/PSL触线
            pbl = g.sell_pbl.get(stock, None)
            pbl_t = g.sell_pbl_t.get(stock, None)
            if pbl is not None and pbl_t is not None:
                try:
                    pbl_f = float(pbl)
                except Exception:
                    pbl_f = None
                if pbl_f is not None and float(snap['r_p']) >= pbl_f:
                    return {'action': 'REBUY', 'destination': None, 'rule': 'L0(PBL触线)', 'time': hm, 'pbl': pbl_f, 'r_p': snap['r_p']}

            psl = g.sell_psl.get(stock, None)
            psl_t = g.sell_psl_t.get(stock, None)
            if psl is not None and psl_t is not None:
                try:
                    psl_f = float(psl)
                except Exception:
                    psl_f = None
                if psl_f is not None and float(snap['r_p']) <= psl_f:
                    src = g.sell_psl_src.get(stock, None) or 'PSL'
                    return {'action': 'SELL_ALL', 'destination': 'pre', 'rule': src, 'time': hm, 'psl': psl_f, 'r_p': snap['r_p']}

            return None

        return None


    @staticmethod
    def _rule_eod_force_exit_by_days_5(stock, today_str):
        """
        临时规则B：盘后持仓天数>=5 强制退出
        action: SELL_ALL
        destination: 'pre'
        """
        if not hasattr(g, 'buy_days') or stock not in g.buy_days:
            return None

        days = g.buy_days.get(stock, 0)
        if days >= 5:
            return {
                'action': 'SELL_ALL',
                'destination': 'pre',
                'rule': 'TMP_FORCE_EXIT_DAYS_5',
                'date': today_str,
                'days': days
            }
        return None

    @staticmethod
    def get_trigger_metrics(stock, now_dt=None):
        """
        在触发卖出相关信号时，抓取当下的指标快照：
        [R-H, T-1-H, R-DIF, T-1-DIF, R-MACD, T-1-MACD] + 本次新增 [R-MA10]

        返回 dict（取不到则为None）：
        {
            'r_h': float or None,
            't1_h': float or None,
            'r_dif': float or None,
            't1_dif': float or None,
            'r_macd': float or None,
            't1_macd': float or None,
            'r_ma10': float or None
        }
        """
        try:
            if now_dt is None:
                now_dt = get_datetime()

            snap = None
            if hasattr(SellRuleEngine, '_get_realtime_snapshot'):
                snap = SellRuleEngine._get_realtime_snapshot(stock, now_dt)

            r_h = r_dif = r_macd = r_ma10 = None
            if snap is not None:
                r_h = snap.get('r_h', None)
                r_dif = snap.get('r_dif', None)
                r_macd = snap.get('r_macd', None)
                r_ma10 = snap.get('r_ma10', None)

            t1_h = None
            if hasattr(g, 'buy_t1_h'):
                t1_h = g.buy_t1_h.get(stock, None)

            t1_dif = None
            if hasattr(g, 'buy_t1_dif'):
                t1_dif = g.buy_t1_dif.get(stock, None)

            t1_macd = None
            if hasattr(g, 'yesterday_macd'):
                t1_macd = g.yesterday_macd.get(stock, None)

            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None

            return {
                'r_h': _to_float(r_h),
                't1_h': _to_float(t1_h),
                'r_dif': _to_float(r_dif),
                't1_dif': _to_float(t1_dif),
                'r_macd': _to_float(r_macd),
                't1_macd': _to_float(t1_macd),
                'r_ma10': _to_float(r_ma10),
            }

        except Exception as e:
            log.info('【触发快照】{} 获取指标失败: {}'.format(stock, str(e)))
            return {
                'r_h': None,
                't1_h': None,
                'r_dif': None,
                't1_dif': None,
                'r_macd': None,
                't1_macd': None,
                'r_ma10': None,
            }


    @staticmethod
    def log_eval_metrics(tag, stock, hm, metrics, extra=None):
        """
        打印统一指标快照（只保留2位小数）
        - 14:55兜底评估：按用户指定格式输出（不输出 extra），覆盖：满仓/全仓/半仓等所有兜底tag
        """
        if metrics is None:
            metrics = {}

        def _fmt2(x):
            try:
                if x is None:
                    return 'None'
                return '{:.2f}'.format(float(x))
            except Exception:
                return str(x)

        # ——关键修复：所有“尾盘兜底评估(*)”都走固定格式，不输出 extra，并且末尾以 | 结尾——
        try:
            if isinstance(tag, str) and tag.startswith('尾盘兜底评估('):
                log.info('{} | stock:{} | hm:{} | '
                         'R-H:{} T-1-H:{} | '
                         'R-DIF:{} T-1-DIF:{} | '
                         'R-MACD:{} T-1-MACD:{} |'.format(
                             tag, stock, hm,
                             _fmt2(metrics.get('r_h')), _fmt2(metrics.get('t1_h')),
                             _fmt2(metrics.get('r_dif')), _fmt2(metrics.get('t1_dif')),
                             _fmt2(metrics.get('r_macd')), _fmt2(metrics.get('t1_macd'))
                         ))
                return
        except Exception:
            pass

        # 其他评估场景：仍可输出 extra（但全部数值2位）
        if extra is None:
            extra = {}
        elif not isinstance(extra, dict):
            extra = {'value': extra}

        # extra 内的数值也尽量统一2位（只处理一层dict）
        extra_fmt = {}
        for k, v in extra.items():
            if isinstance(v, (int, float)):
                extra_fmt[k] = _fmt2(v)
            else:
                extra_fmt[k] = v

        log.info('{} | stock:{} | hm:{} | '
                 'R-H:{} T-1-H:{} | '
                 'R-DIF:{} T-1-DIF:{} | '
                 'R-MACD:{} T-1-MACD:{} | '
                 'extra:{}'.format(
                     tag, stock, hm,
                     _fmt2(metrics.get('r_h')), _fmt2(metrics.get('t1_h')),
                     _fmt2(metrics.get('r_dif')), _fmt2(metrics.get('t1_dif')),
                     _fmt2(metrics.get('r_macd')), _fmt2(metrics.get('t1_macd')),
                     extra_fmt
                 ))


    @staticmethod
    def _is_gap_gt1_between_today_and_xh233_peak_date(stock, now_dt):
        """
        L1-B新增条件：
        触发日 与 “XH-233最高价发生日（XH-233_DATE）” 之间相隔 > 1 个交易日
        即：交易日差值 >= 2 才返回 True
        """
        try:
            if not hasattr(g, 'buy_xh233_date'):
                return False

            peak_date_str = g.buy_xh233_date.get(stock, None)
            if not peak_date_str:
                return False

            today_str = now_dt.strftime('%Y-%m-%d')
            if peak_date_str == today_str:
                return False

            cursor = now_dt
            gap = 0
            for _ in range(400):
                prev_dt = get_previous_trading_date(cursor)
                gap += 1
                prev_str = prev_dt.strftime('%Y-%m-%d')

                if prev_str == peak_date_str:
                    break

                cursor = prev_dt
            else:
                return False

            return gap >= 3

        except Exception as e:
            log.info('【L1-B间隔校验】{} 失败: {}'.format(stock, str(e)))
            return False

# ==================== 5. 触发检测类 ====================
class TriggerChecker:
    """各种触发条件检测"""
    
    @staticmethod
    def check_pre_to_ready(stock):
        """
        检测预备仓→准备仓的触发条件

        条件: 当日分钟级最低价 < 实时10日均线 * 1.005
        额外规则（按02文档）：若股票被标记为“当日不再触发任何操作”，直接返回False
        """
        try:
            current_time = get_datetime()
            today_str = current_time.strftime('%Y-%m-%d')

            # ========= 当日禁触发控制（覆盖“14:55打回预备仓后不再操作”等场景）=========
            if not hasattr(g, 'buy_forbidden_today'):
                g.buy_forbidden_today = {}  # {stock: 'YYYY-MM-DD'}
            if stock in g.buy_forbidden_today and g.buy_forbidden_today[stock] == today_str:
                return False, None
            # =====================================================================

            market_open = today_str + ' 09:30:00'

            # 1. 获取当日分钟数据
            today_data = DataModule.get_today_minute_data(stock, market_open, current_time)

            if today_data is None or len(today_data) == 0:
                return False, None

            today_low = today_data['low'].min()
            current_price = today_data['close'].iloc[-1]

            # 2. 计算实时10日均线
            ma_period = StrategyConfig.MA_PERIOD  # 10
            ma10 = DataModule.calculate_ma(stock, ma_period, current_time, current_price)

            if ma10 is None:
                return False, None

            # 3. 计算触发阈值（增加1.005倍宽容度）
            ma10_threshold = ma10 * 1.005

            # 4. 判断是否触发
            if today_low <= ma10_threshold:
                lookback = StrategyConfig.LOOKBACK_DAYS  # 233
                high_233 = DataModule.get_highest_price(stock, lookback, current_time)

                if high_233 is None:
                    log.info('{} 无法获取233日最高价数据'.format(stock))

                transfer_info = {
                    'time': current_time.strftime('%H:%M'),
                    'low': today_low,
                    'ma10': ma10,
                    'ma10_threshold': ma10_threshold,
                    'current': current_price,
                    'high_233': high_233
                }

                return True, transfer_info

            return False, None

        except Exception as e:
            log.error('{} 预备仓触发检测失败: {}'.format(stock, str(e)))
            return False, None

    @staticmethod
    def check_ready_to_buy(stock):
        """
        检测准备仓→买入仓的触发条件（按 02 文档对齐 + 本次补丁）

        新增定义：
        - PBL = 预买线（buy_line_price）
        - PBL-T：'full'/'half'（本次在“触线买入”时确定；并写入 buy_info）
        - 涨幅检查：IF ((PBL - T-1-C)/T-1-C >= 10%) → 不买入，转回预备仓，清TRG-TIME/买入线，DC按是否创233新高处理，并当日禁触发
        """
        try:
            current_time = get_datetime()
            today_str = current_time.strftime('%Y-%m-%d')

            # ========= 当日禁触发控制 =========
            if not hasattr(g, 'buy_forbidden_today'):
                g.buy_forbidden_today = {}
            if stock in g.buy_forbidden_today and g.buy_forbidden_today[stock] == today_str:
                return False, None

            # 必须有昨日DIF记录（T-1-DIF）
            if not hasattr(g, 'yesterday_dif') or stock not in g.yesterday_dif:
                return False, None
            yesterday_dif = g.yesterday_dif[stock]

            # ========= 获取当前分钟数据（用于R-P、R-H）=========
            market_open = today_str + ' 09:30:00'
            today_data = DataModule.get_today_minute_data(stock, market_open, current_time)
            if today_data is None or len(today_data) == 0:
                return False, None

            current_price = float(today_data['close'].iloc[-1])   # R-P
            today_high = float(today_data['high'].max())          # R-H(截至当前)

            # ========= 步骤1：DIF突发触发（R-DIF >= T-1-DIF）=========
            if not hasattr(g, 'dif_trigger_time'):
                g.dif_trigger_time = {}

            if stock not in g.dif_trigger_time:
                today_dif = DataModule.calculate_realtime_dif(stock, current_time, current_price)  # R-DIF
                if today_dif is None:
                    return False, None

                if float(today_dif) >= float(yesterday_dif):
                    g.dif_trigger_time[stock] = current_time
                    log.info('【DIF触发】{} | 时间:{} | T-1-DIF:{:.2f} | R-DIF:{:.2f}'.format(
                        stock, current_time.strftime('%H:%M'), float(yesterday_dif), float(today_dif)))
                    return False, None  # 触发后等待10分钟

            # ========= 步骤2：触发后10分钟，设置PBL=买入线，并做“涨幅>=10%过滤”=========
            if not hasattr(g, 'buy_line'):
                g.buy_line = {}
            if not hasattr(g, 'buy_line_t'):
                g.buy_line_t = {}

            if stock in g.dif_trigger_time and stock not in g.buy_line:
                trigger_time = g.dif_trigger_time[stock]
                time_diff = (current_time - trigger_time).total_seconds() / 60.0

                if time_diff >= 10:
                    # PBL = R-H * 1.004
                    buy_line_price = today_high * 1.004
                    g.buy_line[stock] = float(buy_line_price)
                    g.buy_line_t[stock] = None  # 先不定型，触线买入时再判定 full/half

                    # --- 获取T-1-C（昨日收盘价）用于涨幅过滤 ---
                    yesterday_dt = None
                    yesterday_close = None
                    try:
                        yesterday_dt = get_previous_trading_date(current_time)
                        ystr = yesterday_dt.strftime('%Y-%m-%d')
                        yday_df = DataModule.get_daily_data(stock, ystr, 5)
                        if yday_df is not None and len(yday_df) > 0:
                            if 'volume' in yday_df.columns:
                                yday_df = yday_df[yday_df['volume'] > 100]
                            if len(yday_df) > 0 and 'close' in yday_df.columns:
                                yesterday_close = float(yday_df['close'].iloc[-1])
                    except Exception:
                        yesterday_close = None

                    # 若拿不到昨收，保守：不做10%过滤，但仍保留买入线
                    if yesterday_close is None or yesterday_close <= 0:
                        log.info('【设置买入线】{} | 时间:{} | R-H:{:.2f} | PBL:{:.2f} | T-1-C获取失败，跳过10%过滤'.format(
                            stock, current_time.strftime('%H:%M'), today_high, float(buy_line_price)))
                        return False, None

                    inc_ratio = (float(buy_line_price) - yesterday_close) / yesterday_close

                    # --- 涨幅>=10%：不买入，转回预备仓，清理TRG-TIME/买入线，DC处理，且当日禁触发 ---
                    if inc_ratio >= 0.10:
                        # DC处理：是否创233日新高（口径：以“昨日为止的H-233”为基准，判断今日是否创出）
                        reset_dc = False
                        try:
                            if yesterday_dt is None:
                                yesterday_dt = get_previous_trading_date(current_time)
                            h233_excl_today = DataModule.get_highest_price(stock, StrategyConfig.LOOKBACK_DAYS, yesterday_dt)
                            if h233_excl_today is not None and today_high >= float(h233_excl_today):
                                reset_dc = True
                        except Exception:
                            reset_dc = False

                        if not hasattr(g, 'decline_count'):
                            g.decline_count = {}

                        if reset_dc:
                            g.decline_count[stock] = 0
                        # else：继承（不改原值）

                        # 转回预备仓
                        if hasattr(g, 'ready_pool') and stock in g.ready_pool:
                            g.ready_pool.remove(stock)
                            if not hasattr(g, 'pre_pool'):
                                g.pre_pool = []
                            if stock not in g.pre_pool:
                                g.pre_pool.append(stock)

                        # 清理 TRG-TIME / 买入线
                        if stock in g.dif_trigger_time:
                            del g.dif_trigger_time[stock]
                        if stock in g.buy_line:
                            del g.buy_line[stock]
                        if hasattr(g, 'buy_line_t') and stock in g.buy_line_t:
                            del g.buy_line_t[stock]

                        # 当日禁触发
                        g.buy_forbidden_today[stock] = today_str

                        log.info('【买入过滤-涨幅>=10%】{} | 时间:{} | T-1-C:{:.2f} | PBL:{:.2f} | 涨幅:{:.2%} => 转回预备仓 | DC:{} | 当日禁触发'.format(
                            stock,
                            current_time.strftime('%H:%M'),
                            yesterday_close,
                            float(buy_line_price),
                            inc_ratio,
                            '重置为0' if reset_dc else '继承'
                        ))
                        return False, None

                    log.info('【设置买入线】{} | 时间:{} | R-H:{:.2f} | PBL:{:.2f} | T-1-C:{:.2f} | 涨幅:{:.2%}'.format(
                        stock, current_time.strftime('%H:%M'), today_high, float(buy_line_price), yesterday_close, inc_ratio))
                    return False, None

            # ========= 步骤3：触线买入（R-P >= PBL）+ PBL-T分流（full/half）=========
            if stock in g.buy_line:
                pbl = float(g.buy_line[stock])

                if current_price >= pbl:
                    # ✅ H-233 口径修正：必须“不含今日”，用 yesterday_dt 截止
                    yesterday_dt = get_previous_trading_date(current_time)
                    high_233 = DataModule.get_highest_price(stock, StrategyConfig.LOOKBACK_DAYS, yesterday_dt)

                    pos_target = None
                    pbl_t = None
                    h233_threshold = None

                    if high_233 is not None:
                        h233_threshold = float(high_233) * 0.97
                        if pbl < h233_threshold:
                            pos_target = 1.0
                            pbl_t = 'full'
                        else:
                            pos_target = 0.5
                            pbl_t = 'half'

                    # 记录到 buy_line_t（给后续链路/审计用）
                    if not hasattr(g, 'buy_line_t'):
                        g.buy_line_t = {}
                    g.buy_line_t[stock] = pbl_t

                    buy_info = {
                        'time': current_time.strftime('%H:%M'),
                        'price': current_price,
                        'buy_line': pbl,
                        'pbl_t': pbl_t,
                        'trigger_time': g.dif_trigger_time[stock].strftime('%H:%M') if stock in g.dif_trigger_time else None,
                        'high_233': high_233,
                        'h233_threshold_097': h233_threshold,
                        'pos_target': pos_target
                    }

                    # 清理该股票的买入相关记录（TRG-TIME / 买入线）
                    if stock in g.dif_trigger_time:
                        del g.dif_trigger_time[stock]
                    if stock in g.buy_line:
                        del g.buy_line[stock]
                    if hasattr(g, 'buy_line_t') and stock in g.buy_line_t:
                        del g.buy_line_t[stock]

                    return True, buy_info

            return False, None

        except Exception as e:
            log.error('{} 准备仓买入检测失败: {}'.format(stock, str(e)))
            return False, None
    
    @staticmethod
    def check_ready_exit(stock, today_str):
        """
        检测准备仓股票是否应该出仓
        ⚠️ 注意：累计下降次数已在 update_all_decline_count 中更新

        条件1: 当日最高价 < 当日10日均线  → 直接出仓（保留原逻辑）
        条件2: DC 动态出仓（新逻辑）：
            IF (DC <= 2) OR (DC == 3 AND V55M == True):
                → 继续持有
            ELSE:
                → 彻底出局（清除所有追踪数据由 transfer_ready_to_exit 做）

        返回: (bool, str, dict)
        """
        try:
            # 获取当日数据
            daily_data = DataModule.get_daily_data(stock, today_str, 1)
            if daily_data is None or len(daily_data) == 0:
                return False, '', None

            today_high = daily_data['high'].iloc[-1]

            # 计算当日10日均线
            ma10 = DataModule.calculate_ma(stock, 10, today_str)

            should_exit = False
            exit_reason = ''

            # ========== 条件1：当日最高价 < 当日10日均线 ==========
            if ma10 is not None and today_high < ma10:
                should_exit = True
                exit_reason = '最高价<MA10'

            # ========== 条件2：DC 动态出仓 ==========
            if not should_exit:
                if not hasattr(g, 'decline_count'):
                    g.decline_count = {}
                if not hasattr(g, 'v55m'):
                    g.v55m = {}

                dc = g.decline_count.get(stock, 0)
                try:
                    dc = int(dc)
                except Exception:
                    dc = 0

                v55m = True if g.v55m.get(stock, False) is True else False

                # 保留条件：DC<=2 或 (DC==3 且 V55M==True)
                if (dc <= 2) or (dc == 3 and v55m):
                    should_exit = False
                else:
                    should_exit = True
                    if dc == 3 and (not v55m):
                        exit_reason = 'DC=3且V55M=False'
                    else:
                        exit_reason = 'DC>=4'

            if should_exit:
                exit_info = {
                    'reason': exit_reason,
                    'time': today_str,
                    'high': today_high,
                    'ma10': ma10 if ma10 else 0,
                    'decline_count': g.decline_count.get(stock, 0),
                    'v55m': g.v55m.get(stock, False),
                    'v55m_date': g.v55m_date.get(stock, None) if hasattr(g, 'v55m_date') else None
                }
                return True, exit_reason, exit_info

            return False, '', None

        except Exception as e:
            log.error('{} 出仓检查失败: {}'.format(stock, str(e)))
            return False, '', None
    
    @staticmethod
    def check_buy_to_sell(stock):
        """
        买入仓 → 卖出/卖半仓/回补 检测（兼容你当前的主循环调用方式）
        - rule 显示为“条件类型”（如 L1-A / L2-4B / ELSE）
        - 若为14:55兜底产生的信号（sig含metrics），日志追加输出指定指标组（两位小数）
        - 当触发条件为 ELSE 时，追加输出 R-MA10
        """
        try:
            if not hasattr(g, 'buy_pool') or stock not in g.buy_pool:
                return False, None, None
            if not hasattr(g, 'buy_pos') or stock not in g.buy_pos:
                return False, None, None

            pos = float(g.buy_pos.get(stock, 1.0))
            days = int(g.buy_days.get(stock, 1)) if hasattr(g, 'buy_days') else 1
            bp = g.buy_bp.get(stock, None) if hasattr(g, 'buy_bp') else None

            sig = SellRuleEngine.check_intraday(stock)
            if sig is None:
                return False, None, None

            action = sig.get('action', None)
            destination = sig.get('destination', None)
            rule = sig.get('rule', 'UNKNOWN')

            if action is None:
                return False, None, None

            now_dt = get_datetime()
            hm = now_dt.strftime('%H:%M')
            today_str = now_dt.strftime('%Y-%m-%d')

            # 两位小数格式
            def _fmt2(v):
                try:
                    if v is None:
                        return 'None'
                    return '{:.2f}'.format(float(v))
                except Exception:
                    return str(v)

            # 尽量拿到执行价
            exec_price = sig.get('price', None)
            if exec_price is None:
                snap = SellRuleEngine._get_realtime_snapshot(stock, now_dt)
                if snap is not None:
                    exec_price = snap.get('r_p', None)

            # 指标串（仅当sig带metrics时输出；ELSE额外加R-MA10）
            metric_str = ''
            mt = sig.get('metrics', None)
            if isinstance(mt, dict):
                metric_str = ' | R-H:{} T-1-H:{} | R-DIF:{} T-1-DIF:{} | R-MACD:{} T-1-MACD:{}'.format(
                    _fmt2(mt.get('r_h', None)), _fmt2(mt.get('t1_h', None)),
                    _fmt2(mt.get('r_dif', None)), _fmt2(mt.get('t1_dif', None)),
                    _fmt2(mt.get('r_macd', None)), _fmt2(mt.get('t1_macd', None))
                )
                if rule == 'ELSE':
                    metric_str += ' | R-MA10:{}'.format(_fmt2(mt.get('r_ma10', None)))

            # ========= 确保收益追踪容器存在 =========
            if not hasattr(g, 'trade_round_qty'):
                g.trade_round_qty = {}
            if not hasattr(g, 'trade_round_avg_cost'):
                g.trade_round_avg_cost = {}
            if not hasattr(g, 'trade_round_invest'):
                g.trade_round_invest = {}
            if not hasattr(g, 'trade_round_realized'):
                g.trade_round_realized = {}
            if not hasattr(g, 'trade_round_logs'):
                g.trade_round_logs = {}
            if not hasattr(g, 'trade_round_result'):
                g.trade_round_result = []
            # =======================================

            # =============== SELL_ALL：不再立即转仓，改为登记到盘后临时列表 ===============
            if action == 'SELL_ALL':
                if destination is None:
                    destination = 'pre'

                info = dict(sig)
                info['pos_from'] = pos
                info['pos_to'] = 0.0
                info['days'] = days
                info['bp'] = bp
                info['price'] = exec_price
                info['time'] = hm
                info['rule'] = rule
                info['action'] = 'SELL_ALL'
                info['destination'] = destination  # 这里保留“原意图”，最终以盘后final_destination为准

                # ====== 先结算本轮收益（POS->0）======
                try:
                    qty_left = float(g.trade_round_qty.get(stock, 0.0))
                    avg_cost = g.trade_round_avg_cost.get(stock, None)
                    invest = float(g.trade_round_invest.get(stock, 0.0))
                    realized = float(g.trade_round_realized.get(stock, 0.0))

                    if exec_price is not None and avg_cost is not None and qty_left > 0:
                        px = float(exec_price)
                        avg = float(avg_cost)
                        realized = realized + (px - avg) * qty_left

                        if stock not in g.trade_round_logs:
                            g.trade_round_logs[stock] = []
                        g.trade_round_logs[stock].append({
                            'type': 'SELL_FINAL',
                            'date': today_str,
                            'qty': qty_left,
                            'price': px,
                            'avg_cost': avg
                        })

                    ret_pct = None
                    if invest > 0:
                        ret_pct = realized / invest * 100.0

                    log.info('【本轮交易结算】{} | 卖出触发:{} | 最终卖出价:{} | 总投入:{} | 已实现盈亏:{} | 收益率:{}%'.format(
                        stock, str(rule), _fmt2(exec_price), _fmt2(invest), _fmt2(realized), _fmt2(ret_pct)
                    ))

                    g.trade_round_result.append({
                        'stock': stock,
                        'end_date': today_str,
                        'rule': rule,
                        'invest': invest,
                        'realized': realized,
                        'ret_pct': ret_pct
                    })
                except Exception:
                    pass

                # ✅ 登记到待盘后去向列表
                PoolManager.add_sell_all_pending(stock, info)

                log.info('【卖出检测】{} | 触发条件:{} | action:SELL_ALL | dest:pending | POS:{}→0.00 | days:{} | bp:{} | price:{}{} | 事件打标:SELL_ALL'.format(
                    stock, rule, _fmt2(pos), str(days), _fmt2(bp), _fmt2(exec_price), metric_str
                ))
                return True, None, info

            # =============== SELL_HALF：卖出半仓（1.0→0.5）==============
            if action == 'SELL_HALF':
                if pos <= 0.5:
                    return False, None, None

                g.buy_pos[stock] = 0.5

                if not hasattr(g, 'pos_half_lock_date'):
                    g.pos_half_lock_date = {}
                g.pos_half_lock_date[stock] = today_str

                # 事件打标
                if not hasattr(g, 'xh_event_flag'):
                    g.xh_event_flag = {}
                if not hasattr(g, 'xh_event_type'):
                    g.xh_event_type = {}
                if not hasattr(g, 'xh_event_date'):
                    g.xh_event_date = {}

                g.xh_event_flag[stock] = today_str
                g.xh_event_type[stock] = 'SELL_HALF'
                g.xh_event_date[stock] = today_str

                info = dict(sig)
                info['pos_from'] = pos
                info['pos_to'] = 0.5
                info['days'] = days
                info['bp'] = bp
                info['price'] = exec_price
                info['time'] = hm
                info['rule'] = rule
                info['action'] = 'SELL_HALF'
                info['destination'] = None

                log.info('【卖出检测】{} | 触发条件:{} | action:SELL_HALF | POS:{}→0.50 | days:{} | bp:{} | price:{}{} | 事件打标:SELL_HALF'.format(
                    stock, rule, _fmt2(pos), str(days), _fmt2(bp), _fmt2(exec_price), metric_str
                ))
                return True, None, info

            # =============== REBUY：回补半仓（0.5→1.0）==============
            if action == 'REBUY':
                if pos >= 1.0:
                    return False, None, None

                # 仓位恢复满仓
                g.buy_pos[stock] = 1.0

                # ✅ 文档要求：回补后 DAYS=3
                if not hasattr(g, 'buy_days'):
                    g.buy_days = {}
                g.buy_days[stock] = 3

                # ✅ 文档要求：记录 REBUY-DATE
                if not hasattr(g, 'rebuy_date'):
                    g.rebuy_date = {}
                g.rebuy_date[stock] = today_str

                # ✅ 文档要求：清除 PBL（回补后不应残留回补线）
                try:
                    if hasattr(g, 'sell_pbl'):
                        g.sell_pbl[stock] = None
                    if hasattr(g, 'sell_pbl_t'):
                        g.sell_pbl_t[stock] = None
                    if hasattr(g, 'sell_pbl_src'):
                        g.sell_pbl_src[stock] = None
                except Exception:
                    pass

                # 事件打标
                if not hasattr(g, 'xh_event_flag'):
                    g.xh_event_flag = {}
                if not hasattr(g, 'xh_event_type'):
                    g.xh_event_type = {}
                if not hasattr(g, 'xh_event_date'):
                    g.xh_event_date = {}

                g.xh_event_flag[stock] = today_str
                g.xh_event_type[stock] = 'REBUY'
                g.xh_event_date[stock] = today_str

                info = dict(sig)
                info['pos_from'] = pos
                info['pos_to'] = 1.0
                info['days'] = days
                info['bp'] = bp
                info['price'] = exec_price
                info['time'] = hm
                info['rule'] = rule
                info['action'] = 'REBUY'
                info['destination'] = None

                log.info('【卖出检测】{} | 触发条件:{} | action:REBUY | POS:{}→1.00 | days:{}→3 | bp:{} | price:{}{}'.format(
                    stock, rule, _fmt2(pos), str(days), _fmt2(bp), _fmt2(exec_price), metric_str
                ))
                return True, None, info

            return False, None, None

        except Exception as e:
            log.error('check_buy_to_sell异常 {}: {}'.format(stock, str(e)))
            return False, None, None


# ==================== 6. 仓位管理类 ====================
class PoolManager:
    """管理各个仓位的股票流转"""
    
    @staticmethod
    def add_to_pre_pool(stock_list):
        """
        添加股票到预备仓（去重）
        返回: (int, list) - (新增数量, 新增股票列表)

        新增：初始化 V55M（成交量是否创过去55个交易日新高）
        - 若当日成交量 > 截至昨日的近55日最大量，则 V55M=True（并记录日期）
        """
        if stock_list is None:
            log.info('传入的股票列表为None')
            return 0, []

        if not isinstance(stock_list, list):
            log.info('传入的不是列表类型')
            return 0, []

        # 兼容：确保V55M容器存在
        if not hasattr(g, 'v55m'):
            g.v55m = {}
        if not hasattr(g, 'v55m_date'):
            g.v55m_date = {}

        added_count = 0
        newly_added = []
        today_str = get_datetime().strftime('%Y-%m-%d')

        for stock in stock_list:
            if stock not in g.pre_pool and stock not in g.ready_pool and stock not in g.buy_pool:
                g.pre_pool.append(stock)
                newly_added.append(stock)
                added_count += 1

                # ========== 初始化：记录进入预备仓的日期和当日数据 ==========
                g.entry_date[stock] = today_str
                g.decline_count[stock] = 0

                # 获取当日数据作为累计下降判断的初始基准
                data = get_price(
                    [stock], None, today_str, '1d',
                    ['close', 'high', 'volume'],
                    False, 'pre', 120
                )

                if data is not None and stock in data:
                    df = data[stock]

                    # 过滤真实交易日
                    valid_days = df[df['volume'] > 100]

                    if len(valid_days) >= 26:
                        # 记录当日最高价
                        today_high = valid_days['high'].iloc[-1]
                        g.prev_high[stock] = today_high

                        # 计算并记录当日DIF
                        today_closes = valid_days['close']
                        ema12 = today_closes.ewm(span=12, adjust=False).mean().iloc[-1]
                        ema26 = today_closes.ewm(span=26, adjust=False).mean().iloc[-1]
                        today_dif = ema12 - ema26
                        g.prev_dif[stock] = today_dif

                    # ========== 新增：初始化V55M ==========
                    # 判定口径：当日成交量 > 截至昨日的过去55个交易日最大成交量（严格大于才算“创”）
                    try:
                        if len(valid_days) >= 2:
                            today_vol = float(valid_days['volume'].iloc[-1])

                            # 截至昨日（不含当日）的历史
                            hist = valid_days.iloc[:-1]
                            if len(hist) > 0:
                                hist55 = hist.tail(55)
                                prev_max = float(hist55['volume'].max())
                                if today_vol > prev_max:
                                    g.v55m[stock] = True
                                    g.v55m_date[stock] = today_str
                                else:
                                    g.v55m[stock] = False
                            else:
                                # 无历史可比：保守为False
                                g.v55m[stock] = False
                        else:
                            g.v55m[stock] = False
                    except Exception as e:
                        g.v55m[stock] = False
                        log.info('【V55M初始化】{} 失败: {}'.format(stock, str(e)))
                    # =====================================
                # ================================================================

        return added_count, newly_added
    
    @staticmethod
    def transfer_pre_to_ready(stock, transfer_info):
        """
        预备仓 → 准备仓

        按 02 文档阶段3：V55M 的窗口口径
        - 窗口起点：进入准备仓日往前推5个交易日
        - 窗口终点：DC=3当日（由盘后更新DC时判断，但V55M可以提前监控）
        """
        if stock in g.pre_pool:
            g.pre_pool.remove(stock)
            g.ready_pool.append(stock)
            g.transfer_info[stock] = transfer_info

            current_time = get_datetime()
            today_str = current_time.strftime('%Y-%m-%d')

            # ========== 记录进入准备仓日期（用于V55M窗口）==========
            if not hasattr(g, 'ready_entry_date'):
                g.ready_entry_date = {}
            g.ready_entry_date[stock] = today_str

            # 窗口起点：进入准备仓前5个交易日
            if not hasattr(g, 'v55m_start_date'):
                g.v55m_start_date = {}

            start_dt = current_time
            for _ in range(5):
                start_dt = get_previous_trading_date(start_dt)
            start_str = start_dt.strftime('%Y-%m-%d')
            g.v55m_start_date[stock] = start_str
            # ======================================================

            # ========== 只记录昨日DIF（用于买入触发）==========
            # 获取历史数据
            data = get_price(
                [stock], None, current_time, '1d',
                ['close', 'volume'],
                False, 'pre', 100
            )

            if data is not None and stock in data:
                df = data[stock]

                # 过滤真实交易日
                valid_days = df[df['volume'] > 100]

                if len(valid_days) >= 27:
                    # 记录昨日DIF（用于买入触发）
                    yesterday_closes = valid_days['close'].iloc[:-1]

                    if len(yesterday_closes) >= 26:
                        ema12 = yesterday_closes.ewm(span=12, adjust=False).mean().iloc[-1]
                        ema26 = yesterday_closes.ewm(span=26, adjust=False).mean().iloc[-1]
                        yesterday_dif = ema12 - ema26
                        g.yesterday_dif[stock] = yesterday_dif
                        log.info('  → 记录昨日DIF: {:.4f}'.format(yesterday_dif))
            # ==================================================

            # ========== 入准备仓时：回溯计算窗口内是否已出现V55M ==========
            # 口径：窗口内任意一天 volume > 该日之前55个交易日最大volume（严格大于）
            if not hasattr(g, 'v55m'):
                g.v55m = {}
            if not hasattr(g, 'v55m_date'):
                g.v55m_date = {}

            try:
                # 拉取足够覆盖：窗口起点前55天的历史（这里取 140 根日线做冗余）
                df = DataModule.get_daily_data(stock, today_str, 140)
                if df is not None and len(df) > 0:
                    if 'volume' in df.columns:
                        df = df[df['volume'] > 100]
                    df = df.dropna()

                v55m_hit = False
                v55m_first_date = None

                if df is not None and len(df) >= 60:
                    # df的索引一般是日期（Timestamp），这里按行序遍历
                    for i in range(len(df)):
                        idx = df.index[i]
                        try:
                            dstr = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
                        except Exception:
                            dstr = None

                        if dstr is None:
                            continue

                        # 只看窗口内日期
                        if dstr < start_str or dstr > today_str:
                            continue

                        # 需要“该日之前”至少55个交易日
                        if i < 55:
                            continue

                        today_vol = float(df['volume'].iloc[i])
                        prev55_max = float(df['volume'].iloc[i-55:i].max())

                        if today_vol > prev55_max:
                            v55m_hit = True
                            v55m_first_date = dstr
                            break

                if v55m_hit:
                    g.v55m[stock] = True
                    g.v55m_date[stock] = v55m_first_date
                else:
                    g.v55m[stock] = False

            except Exception as e:
                g.v55m[stock] = False
                log.info('【V55M窗口回溯】{} 失败: {}'.format(stock, str(e)))
            # ==========================================================

            high_233 = transfer_info.get('high_233', 0)
            high_233_str = '{:.2f}'.format(high_233) if high_233 else 'N/A'

            log.info('【预备仓→准备仓】{} | 时间:{} | 最低:{:.2f} | MA10:{:.2f} | 233日最高:{} | V55M窗口起点:{} | V55M:{}({})'.format(
                stock,
                transfer_info.get('time', ''),
                transfer_info.get('low', 0),
                transfer_info.get('ma10', 0),
                high_233_str,
                start_str,
                str(g.v55m.get(stock, False)),
                str(g.v55m_date.get(stock, None))
            ))

    
    @staticmethod
    def transfer_ready_to_buy(stock, buy_info):
        """
        准备仓 → 买入仓（按 02 文档：买入后初始化）
        并新增：X-H / XH-233 使用“事件锚点”语义，买入事件打标，盘后统一回填。

        本次补丁：落地 PBL / PBL-T
        - PBL = buy_info['buy_line']
        - PBL-T = buy_info['pbl_t'] ('full'/'half')

        新增：初始化本轮交易收益追踪（trade_round_*）
        - 投入金额按“仓位比例”记账：full=1.0, half=0.5
        - 仅在 POS=0 时结算并打印整轮收益
        """
        if stock not in g.ready_pool:
            return

        # ========= 确保承载结构存在 =========
        if not hasattr(g, 'buy_pos'):
            g.buy_pos = {}
        if not hasattr(g, 'buy_days'):
            g.buy_days = {}
        if not hasattr(g, 'buy_bp'):
            g.buy_bp = {}
        if not hasattr(g, 'buy_time'):
            g.buy_time = {}
        if not hasattr(g, 'buy_xh233'):
            g.buy_xh233 = {}
        if not hasattr(g, 'buy_xh233_date'):
            g.buy_xh233_date = {}
        if not hasattr(g, 'buy_xh'):
            g.buy_xh = {}
        if not hasattr(g, 'buy_t1_h'):
            g.buy_t1_h = {}
        if not hasattr(g, 'buy_t1_dif'):
            g.buy_t1_dif = {}
        if not hasattr(g, 'buy_dc'):
            g.buy_dc = {}

        # ——新增：买入时落地PBL/PBL-T（为回补/半仓链路做准备）——
        if not hasattr(g, 'buy_pbl'):
            g.buy_pbl = {}       # {stock: PBL}
        if not hasattr(g, 'buy_pbl_t'):
            g.buy_pbl_t = {}     # {stock: 'full'/'half'/None}

        # ========= 事件锚点标记 =========
        if not hasattr(g, 'xh_event_flag'):
            g.xh_event_flag = {}
        if not hasattr(g, 'xh_event_type'):
            g.xh_event_type = {}
        if not hasattr(g, 'xh_event_date'):
            g.xh_event_date = {}
        # =====================================

        current_time = get_datetime()
        today_str = current_time.strftime('%Y-%m-%d')

        # ========= 计算T-1-H（昨日最高）=========
        t1_high = None
        yesterday_dt = None
        try:
            yesterday_dt = get_previous_trading_date(current_time)
            yesterday_str = yesterday_dt.strftime('%Y-%m-%d')
            yday_df = DataModule.get_daily_data(stock, yesterday_str, 5)
            if yday_df is not None and len(yday_df) > 0:
                if 'volume' in yday_df.columns:
                    yday_df = yday_df[yday_df['volume'] > 100]
                if len(yday_df) > 0:
                    t1_high = float(yday_df['high'].iloc[-1])
        except Exception as e:
            log.info('【买入初始化】{} 获取T-1-H失败: {}'.format(stock, str(e)))

        # ========= T-1-DIF（昨日DIF）=========
        t1_dif = g.yesterday_dif.get(stock, None)
        try:
            if t1_dif is not None:
                t1_dif = float(t1_dif)
        except Exception:
            t1_dif = None

        # ========= DC继承（下降次数）=========
        try:
            dc = g.decline_count.get(stock, 0)
        except Exception:
            dc = 0

        # ========= POS目标（全仓/半仓）=========
        pos_target = buy_info.get('pos_target', None)
        if pos_target is None:
            pos_target = 1.0

        # ========= 关键字段（B-P、时间）=========
        buy_price = buy_info.get('price', None)
        buy_time = buy_info.get('time', '')

        # ========= PBL / PBL-T（新增）=========
        pbl = buy_info.get('buy_line', None)
        pbl_t = buy_info.get('pbl_t', None)

        # ========= XH-233 + XH-233_DATE（事件日前233日最高，不含事件日；截止“昨天”）=========
        xh_233 = None
        xh_233_date = None
        try:
            if yesterday_dt is None:
                yesterday_dt = get_previous_trading_date(current_time)
            xh_233, xh_233_date = DataModule.get_highest_price_with_date(
                stock, StrategyConfig.LOOKBACK_DAYS, yesterday_dt
            )
        except Exception as e:
            log.info('【买入初始化】{} 获取XH-233(截止昨日)失败: {}'.format(stock, str(e)))

        # ========= 转仓动作 =========
        g.ready_pool.remove(stock)
        if stock not in g.buy_pool:
            g.buy_pool.append(stock)

        # ========= 写入审计记录 =========
        g.buy_info[stock] = buy_info

        # ========= 写入结构化状态 =========
        g.buy_pos[stock] = float(pos_target)
        g.buy_days[stock] = 1
        g.buy_bp[stock] = buy_price
        g.buy_time[stock] = buy_time

        # ——新增：落地 PBL / PBL-T
        g.buy_pbl[stock] = pbl
        g.buy_pbl_t[stock] = pbl_t

        # 事件锚点：X-H 盘后回填；XH-233 / XH-233_DATE 可先写入
        g.buy_xh[stock] = None
        g.buy_xh233[stock] = xh_233
        g.buy_xh233_date[stock] = xh_233_date

        g.buy_t1_h[stock] = t1_high
        g.buy_t1_dif[stock] = t1_dif
        g.buy_dc[stock] = int(dc)

        # ========= 关键事件打标：BUY（盘后刷新 X-H / XH-233 / XH-233_DATE）=========
        g.xh_event_flag[stock] = today_str
        g.xh_event_type[stock] = 'BUY'
        # ============================================================

        # ========= 新增：本轮交易收益追踪初始化 =========
        # trade_round_qty：当前持仓比例（0/0.5/1.0）
        # trade_round_avg_cost：加权成本（按比例）
        # trade_round_invest：累计投入金额（按比例）
        # trade_round_realized：累计已实现盈亏（按比例）
        if not hasattr(g, 'trade_round_qty'):
            g.trade_round_qty = {}
        if not hasattr(g, 'trade_round_avg_cost'):
            g.trade_round_avg_cost = {}
        if not hasattr(g, 'trade_round_invest'):
            g.trade_round_invest = {}
        if not hasattr(g, 'trade_round_realized'):
            g.trade_round_realized = {}
        if not hasattr(g, 'trade_round_logs'):
            g.trade_round_logs = {}   # {stock: [dict,...]} 可选：留审计
        if not hasattr(g, 'trade_round_result'):
            g.trade_round_result = []  # 可选：整轮结束后汇总

        try:
            qty = float(pos_target)
        except Exception:
            qty = 1.0

        bp0 = None
        try:
            bp0 = float(buy_price) if buy_price is not None else None
        except Exception:
            bp0 = None

        if bp0 is not None and bp0 > 0:
            g.trade_round_qty[stock] = qty
            g.trade_round_avg_cost[stock] = bp0
            g.trade_round_invest[stock] = qty * bp0
            g.trade_round_realized[stock] = 0.0
            g.trade_round_logs[stock] = [{
                'type': 'BUY_INIT',
                'date': today_str,
                'qty': qty,
                'price': bp0
            }]
        else:
            # 若买入价缺失：仍初始化结构，避免后续KeyError
            g.trade_round_qty[stock] = qty
            g.trade_round_avg_cost[stock] = None
            g.trade_round_invest[stock] = 0.0
            g.trade_round_realized[stock] = 0.0
            g.trade_round_logs[stock] = [{
                'type': 'BUY_INIT',
                'date': today_str,
                'qty': qty,
                'price': None
            }]

        # ========= 兜底清理（TRG-TIME / 买入线）=========
        if hasattr(g, 'dif_trigger_time') and stock in g.dif_trigger_time:
            del g.dif_trigger_time[stock]
        if hasattr(g, 'buy_line') and stock in g.buy_line:
            del g.buy_line[stock]
        if hasattr(g, 'buy_line_t') and stock in g.buy_line_t:
            del g.buy_line_t[stock]

        log.info('【准备仓→买入仓】{} | POS:{} | DAYS:1 | B-P:{} | 买入时间:{} | PBL:{} | PBL-T:{} | XH-233:{} | XH-233_DATE:{} | T-1-H:{} | T-1-DIF:{} | DC:{} | 事件打标:BUY'.format(
            stock,
            g.buy_pos[stock],
            str(buy_price),
            buy_time,
            str(pbl),
            str(pbl_t),
            str(xh_233),
            str(xh_233_date),
            str(t1_high),
            str(t1_dif),
            dc
        ))

    @staticmethod
    def transfer_ready_to_exit(stock, exit_info):
        """
        准备仓 → 直接出仓
        STEP7：走统一清理口径，防止幽灵状态
        """
        # 先记录exit_info（便于审计）
        try:
            if not hasattr(g, 'exit_info'):
                g.exit_info = {}
            g.exit_info[stock] = exit_info
        except Exception:
            pass

        # 统一清理（会自动从 ready_pool 移除）
        PoolManager._cleanup_stock_on_exit(stock)
    
    @staticmethod
    def transfer_buy_to_destination(stock, destination, sell_info):
        """
        买入仓 → 根据规则决定去向

        destination: 'pre' / 'ready' / 'exit'

        新增：当 POS=0（SELL_ALL）时，结算本轮交易收益（trade_round_*）
        """
        if stock not in g.buy_pool:
            return

        # ========= 收益结算：仅在 SELL_ALL / POS->0 时触发 =========
        try:
            action = sell_info.get('action', None)
        except Exception:
            action = None

        if action == 'SELL_ALL':
            # 取本次卖出价（尽量从sell_info里取）
            exec_price = sell_info.get('price', None)
            if exec_price is None:
                exec_price = sell_info.get('r_p', None)

            if exec_price is None:
                try:
                    now_dt = get_datetime()
                    today_str = now_dt.strftime('%Y-%m-%d')
                    market_open = today_str + ' 09:30:00'
                    df = DataModule.get_today_minute_data(stock, market_open, now_dt)
                    if df is not None and len(df) > 0:
                        exec_price = float(df['close'].iloc[-1])
                except Exception:
                    exec_price = None

            # 容器确保存在
            if not hasattr(g, 'trade_round_qty'):
                g.trade_round_qty = {}
            if not hasattr(g, 'trade_round_avg_cost'):
                g.trade_round_avg_cost = {}
            if not hasattr(g, 'trade_round_invest'):
                g.trade_round_invest = {}
            if not hasattr(g, 'trade_round_realized'):
                g.trade_round_realized = {}
            if not hasattr(g, 'trade_round_logs'):
                g.trade_round_logs = {}
            if not hasattr(g, 'trade_round_result'):
                g.trade_round_result = []

            # 计算“最后一笔卖出”贡献，并结算收益率
            try:
                qty_left = float(g.trade_round_qty.get(stock, 0.0))
                avg_cost = g.trade_round_avg_cost.get(stock, None)
                invest = float(g.trade_round_invest.get(stock, 0.0))
                realized = float(g.trade_round_realized.get(stock, 0.0))

                if exec_price is not None and avg_cost is not None and qty_left > 0:
                    px = float(exec_price)
                    avg = float(avg_cost)
                    realized = realized + (px - avg) * qty_left

                    # 记录日志
                    today_str = get_datetime().strftime('%Y-%m-%d')
                    if stock not in g.trade_round_logs:
                        g.trade_round_logs[stock] = []
                    g.trade_round_logs[stock].append({
                        'type': 'SELL_FINAL',
                        'date': today_str,
                        'qty': qty_left,
                        'price': px,
                        'avg_cost': avg
                    })

                # 收益率：已实现盈亏 / 总投入
                ret_pct = None
                if invest > 0:
                    ret_pct = realized / invest * 100.0

                # 输出（两位小数）
                def _fmt2(x):
                    try:
                        if x is None:
                            return 'None'
                        return '{:.2f}'.format(float(x))
                    except Exception:
                        return str(x)

                log.info('【本轮交易结算】{} | 卖出触发:{} | 最终卖出价:{} | 总投入:{} | 已实现盈亏:{} | 收益率:{}%'.format(
                    stock,
                    str(sell_info.get('rule', 'UNKNOWN')),
                    _fmt2(exec_price),
                    _fmt2(invest),
                    _fmt2(realized),
                    _fmt2(ret_pct)
                ))

                g.trade_round_result.append({
                    'stock': stock,
                    'end_date': get_datetime().strftime('%Y-%m-%d'),
                    'rule': sell_info.get('rule', 'UNKNOWN'),
                    'final_price': exec_price,
                    'invest': invest,
                    'realized': realized,
                    'ret_pct': ret_pct
                })

            except Exception as e:
                log.info('【本轮交易结算】{} 计算失败: {}'.format(stock, str(e)))

            # 清理本轮追踪（避免下一轮污染）
            try:
                if stock in g.trade_round_qty:
                    del g.trade_round_qty[stock]
                if stock in g.trade_round_avg_cost:
                    del g.trade_round_avg_cost[stock]
                if stock in g.trade_round_invest:
                    del g.trade_round_invest[stock]
                if stock in g.trade_round_realized:
                    del g.trade_round_realized[stock]
                if stock in g.trade_round_logs:
                    del g.trade_round_logs[stock]
            except Exception:
                pass
        # ============================================================

        g.buy_pool.remove(stock)

        if destination == 'pre':
            g.pre_pool.append(stock)
            log.info('【买入仓→预备仓】{}'.format(stock))
        elif destination == 'ready':
            g.ready_pool.append(stock)
            log.info('【买入仓→准备仓】{}'.format(stock))
        elif destination == 'exit':
            log.info('【买入仓→出仓】{}'.format(stock))

        g.sell_info[stock] = sell_info

    @staticmethod
    def add_sell_all_pending(stock, sell_info):
        """
        STEP2/STEP6：SELL_ALL 不再立即决定去向。
        盘中/盘后一旦发生 SELL_ALL：
        - 先从 buy_pool 移除（避免当天继续触发其它买入仓逻辑）
        - 仓位写0
        - 清理当日半仓锁、以及所有卖出/回补线状态（避免残留污染后续）
        - 记录到 sell_all_pending
        - 不加入 pre/ready/exit，真正去向在 after_trading 用 T-0-L/T-0-H/MA-10/DC 决策
        """
        if not hasattr(g, 'sell_all_pending'):
            g.sell_all_pending = {}

        # 兜底：记录卖出信息
        try:
            info = dict(sell_info) if sell_info is not None else {}
        except Exception:
            info = {}

        # 从买入仓移除，防止继续盘中触发
        if hasattr(g, 'buy_pool') and stock in g.buy_pool:
            try:
                g.buy_pool.remove(stock)
            except Exception:
                pass

        # 仓位写0（逻辑上已全卖）
        if hasattr(g, 'buy_pos') and stock in g.buy_pos:
            g.buy_pos[stock] = 0.0

        # SELL_ALL 后半仓锁无意义，清理
        if hasattr(g, 'pos_half_lock_date') and stock in g.pos_half_lock_date:
            try:
                del g.pos_half_lock_date[stock]
            except Exception:
                pass

        # ——STEP6补齐：SELL_ALL 后清理卖出/回补线，避免残留污染后续日志与判断——
        try:
            if hasattr(g, 'sell_psl'):
                g.sell_psl[stock] = None
            if hasattr(g, 'sell_psl_t'):
                g.sell_psl_t[stock] = None
            if hasattr(g, 'sell_psl_src'):
                g.sell_psl_src[stock] = None

            if hasattr(g, 'sell_pbl'):
                g.sell_pbl[stock] = None
            if hasattr(g, 'sell_pbl_t'):
                g.sell_pbl_t[stock] = None
            if hasattr(g, 'sell_pbl_src'):
                g.sell_pbl_src[stock] = None
        except Exception:
            pass

        # 登记到临时列表
        g.sell_all_pending[stock] = info

        # 保留一份 sell_info 供审计
        if hasattr(g, 'sell_info'):
            g.sell_info[stock] = info

        log.info('【SELL_ALL登记-待盘后去向】{} | 触发条件:{} | price:{} | 说明:盘后用T-0-L/T-0-H/MA-10/DC决定去向'.format(
            stock,
            str(info.get('rule', info.get('trigger', ''))),
            str(info.get('price', info.get('r_p', None)))
        ))

    @staticmethod
    def process_sell_all_pending_after_close(today_str):
        """
        STEP2/STEP6：盘后统一执行“卖出后去向判断（POS=0）”
        使用盘后数据：
        - T-0-L、T-0-H（当日日线）
        - MA-10（盘后10日均线，收盘价口径）
        - DC（decline_count）

        逻辑（与你给出的文档一致）：
        IF (DC <= 2):
            IF (T-0-L > MA-10)                  -> 去预备仓（DC保留）
            ELIF (T-0-L <= MA-10 <= T-0-H)      -> 去准备仓（DC保留）
            ELIF (T-0-H < MA-10)                -> 彻底离场（清追踪）
        ELSE (DC > 2):
            -> 彻底离场（清追踪）
        """
        if not hasattr(g, 'sell_all_pending'):
            return

        pending_stocks = list(g.sell_all_pending.keys())
        if not pending_stocks:
            return

        moved_pre = []
        moved_ready = []
        moved_exit = []

        for stock in pending_stocks:
            info = g.sell_all_pending.get(stock, {})

            # 取DC（若缺失，按高风险处理为>2，直接出仓）
            dc = 999
            if hasattr(g, 'decline_count'):
                try:
                    dc = int(g.decline_count.get(stock, 999))
                except Exception:
                    dc = 999

            # 取当日日线 T-0-L / T-0-H
            t0_l = None
            t0_h = None
            daily = DataModule.get_daily_data(stock, today_str, 1)
            if daily is None or len(daily) == 0:
                dest = 'exit'
            else:
                try:
                    row = daily.iloc[-1]
                    t0_l = float(row['low'])
                    t0_h = float(row['high'])
                except Exception:
                    t0_l, t0_h = None, None

                ma10 = DataModule.calculate_ma(stock, 10, today_str)

                dest = 'exit'
                if ma10 is None or t0_l is None or t0_h is None:
                    dest = 'exit'
                else:
                    ma10 = float(ma10)
                    if dc <= 2:
                        if t0_l > ma10:
                            dest = 'pre'
                        elif t0_l <= ma10 <= t0_h:
                            dest = 'ready'
                        elif t0_h < ma10:
                            dest = 'exit'
                    else:
                        dest = 'exit'

            # 执行去向
            if dest == 'pre':
                if stock not in g.pre_pool:
                    g.pre_pool.append(stock)
                moved_pre.append(stock)

            elif dest == 'ready':
                if stock not in g.ready_pool:
                    g.ready_pool.append(stock)
                moved_ready.append(stock)

            else:
                moved_exit.append(stock)

                # ✅ STEP7：exit 统一清理（防幽灵状态）
                PoolManager._cleanup_stock_on_exit(stock)

            # 记录最终去向到 sell_info 便于审计
            try:
                if hasattr(g, 'sell_info'):
                    if stock not in g.sell_info:
                        g.sell_info[stock] = {}
                    g.sell_info[stock]['final_destination'] = dest
                    g.sell_info[stock]['t0_l'] = t0_l
                    g.sell_info[stock]['t0_h'] = t0_h
                    g.sell_info[stock]['dc'] = dc
            except Exception:
                pass

            # 从 pending 删除
            try:
                if hasattr(g, 'sell_all_pending') and stock in g.sell_all_pending:
                    del g.sell_all_pending[stock]
            except Exception:
                pass

            log.info('【卖出后去向判定-盘后】{} | DC:{} | T-0-L:{} | T-0-H:{} | 去向:{}'.format(
                stock, str(dc), str(t0_l), str(t0_h), dest
            ))

        if moved_pre or moved_ready or moved_exit:
            log.info('【SELL_ALL盘后去向汇总】日期:{} | 去预备仓:{} | 去准备仓:{} | 出仓:{} | 样本pre:{} ready:{} exit:{}'.format(
                today_str,
                len(moved_pre),
                len(moved_ready),
                len(moved_exit),
                moved_pre[:10],
                moved_ready[:10],
                moved_exit[:10]
            ))


    @staticmethod
    def process_ready_pool_exit(today_str):
        """
        批量检查并处理准备仓出仓
        
        返回: list of tuple (stock, reason, time)
        """
        exit_stocks = []
        
        for stock in g.ready_pool[:]:  # 使用切片避免迭代时修改
            should_exit, reason, exit_info = TriggerChecker.check_ready_exit(stock, today_str)
            
            if should_exit:
                PoolManager.transfer_ready_to_exit(stock, exit_info)
                exit_stocks.append((stock, reason, today_str))
        
        return exit_stocks
    
    @staticmethod
    def get_pool_status():
        """
        获取各仓位统计信息
        返回: dict
        """
        return {
            'pre_count': len(g.pre_pool),
            'ready_count': len(g.ready_pool),
            'buy_count': len(g.buy_pool),
            'pre_list': g.pre_pool,
            'ready_list': g.ready_pool,
            'buy_list': g.buy_pool
        }

    @staticmethod
    def record_sell_1455_log(stock, now_dt, hm, pos, days, bp, rule, metrics, extra=None):
        """
        14:55兜底评估专用日志记录（修复缺失方法导致的 AttributeError）
        - 仅做记录，不改变任何交易状态，避免污染逻辑
        - 复用 SellRuleEngine.log_eval_metrics 的格式化输出
        """
        try:
            tag = '尾盘兜底评估({})'.format(str(rule))

            # 统一走卖出引擎的指标打印格式（两位小数、末尾 |）
            SellRuleEngine.log_eval_metrics(
                tag=tag,
                stock=stock,
                hm=hm,
                metrics=metrics,
                extra=extra
            )

            # 额外补充一行简要上下文（可删，不影响逻辑）
            log.info('【1455兜底评估记录】{} | hm:{} | POS:{} | DAYS:{} | BP:{}'.format(
                stock, str(hm), str(pos), str(days), str(bp)
            ))

        except Exception as e:
            log.info('【1455兜底评估记录】{} 写日志失败: {}'.format(stock, str(e)))

    @staticmethod
    def reset_daily_buy_tracking():
        """
        重置当日买入跟踪变量
        每日盘前调用
        """
        g.dif_trigger_time = {}
        g.buy_line = {}
    
    @staticmethod
    def record_yesterday_dif(today_str):
        """
        记录准备仓所有股票的昨日DIF值
        每日盘后调用
        
        返回: int - 成功记录的股票数量
        """
        success_count = 0
        
        for stock in g.ready_pool:
            dif = DataModule.calculate_macd_dif(stock, today_str)
            if dif is not None:
                g.yesterday_dif[stock] = dif
                success_count += 1
        
        return success_count

    @staticmethod
    def update_all_decline_count(today_str):
        """
        更新所有预备仓和准备仓股票的累计下降次数
        每日盘后调用
        
        返回: dict - 各股票的累计次数
        """
        decline_summary = {}
        
        # 检查预备仓和准备仓的所有股票
        all_stocks = g.pre_pool + g.ready_pool
        
        for stock in all_stocks:
            # 检查是否有前一日数据
            if stock not in g.prev_high or stock not in g.prev_dif:
                continue
            
            # 获取当日数据
            daily_data = DataModule.get_daily_data(stock, today_str, 1)
            if daily_data is None or len(daily_data) == 0:
                continue
            
            today_high = daily_data['high'].iloc[-1]
            today_dif = DataModule.calculate_macd_dif(stock, today_str)
            
            if today_dif is None:
                continue
            
            # 检查是否创233日新高（重置计数条件）
            lookback = StrategyConfig.LOOKBACK_DAYS  # 233
            high_233 = DataModule.get_highest_price(stock, lookback, today_str)
            
            if high_233 is not None and today_high >= high_233:
                # 创233日新高，重置计数
                g.decline_count[stock] = 0
                decline_summary[stock] = {'count': 0, 'reason': '创233日新高'}
            else:
                # 未创新高，检查是否同时下降
                if today_dif < g.prev_dif[stock] and today_high < g.prev_high[stock]:
                    # 同时下降，累计计数+1
                    g.decline_count[stock] = g.decline_count.get(stock, 0) + 1
                    decline_summary[stock] = {'count': g.decline_count[stock], 'reason': '同时下降'}
                else:
                    # 不是同时下降，保持原计数
                    decline_summary[stock] = {'count': g.decline_count.get(stock, 0), 'reason': '保持'}
            
            # 更新前一日数据（用于明天对比）
            g.prev_high[stock] = today_high
            g.prev_dif[stock] = today_dif
        
        return decline_summary

    @staticmethod
    def update_v55m_flags(today_str):
        """
        盘后增量更新 V55M（严格按02文档窗口口径）
        仅对 ready_pool 内、且 V55M=False 的股票：
        - 若 today_str >= v55m_start_date[stock] 才允许开始监控
        - 判定口径：今日成交量 > 今日之前55个交易日最大成交量（严格大于）
        一旦 True：永久保持，并记录首次日期 v55m_date
        """
        if not hasattr(g, 'v55m'):
            g.v55m = {}
        if not hasattr(g, 'v55m_date'):
            g.v55m_date = {}
        if not hasattr(g, 'v55m_start_date'):
            g.v55m_start_date = {}

        if not hasattr(g, 'ready_pool') or not g.ready_pool:
            return 0

        updated = 0

        for stock in list(g.ready_pool):
            # 已经True的不再计算
            if g.v55m.get(stock, False) is True:
                continue

            start_str = g.v55m_start_date.get(stock, None)
            # 若缺失窗口起点（异常情况），保守：不更新
            if not start_str:
                continue

            # 未到窗口期，不允许打标
            if today_str < start_str:
                continue

            try:
                # 取至少 120 根，确保“今日之前55日”有足够数据
                df = DataModule.get_daily_data(stock, today_str, 160)
                if df is None or len(df) == 0:
                    continue

                if 'volume' in df.columns:
                    df = df[df['volume'] > 100]
                if df is None or len(df) < 60:
                    continue

                # df最后一行视为今天
                today_vol = float(df['volume'].iloc[-1])

                hist = df.iloc[:-1]
                if hist is None or len(hist) < 55:
                    continue

                prev55_max = float(hist.tail(55)['volume'].max())

                if today_vol > prev55_max:
                    g.v55m[stock] = True
                    g.v55m_date[stock] = today_str
                    updated += 1
                    log.info('【V55M更新(窗口内)】{} | 日期:{} | 窗口起点:{} | 今日量:{:.2f} | 前55日最大量:{:.2f} => V55M=True'.format(
                        stock, today_str, start_str, today_vol, prev55_max
                    ))
                else:
                    g.v55m[stock] = False

            except Exception as e:
                log.info('【V55M更新(窗口内)】{} 异常: {}'.format(stock, str(e)))

        if updated > 0:
            log.info('【盘后V55M增量更新】日期:{} | 新增True:{}只'.format(today_str, updated))
        return updated

    @staticmethod
    def apply_action_from_sell_signal(stock, sig):
        """
        统一执行卖出系统动作入口（集中化，避免规则各自改状态造成污染）

        新增：当 SELL_ALL 导致 POS=0 时，结算本轮交易收益（trade_round_*）
        """
        try:
            action = sig.get('action', None)
            destination = sig.get('destination', None)
            rule = sig.get('rule', 'UNKNOWN_RULE')

            if action is None:
                return False, None, None

            # 半仓锁容器
            if not hasattr(g, 'pos_half_lock_date'):
                g.pos_half_lock_date = {}

            # ========= 确保收益追踪容器存在 =========
            if not hasattr(g, 'trade_round_qty'):
                g.trade_round_qty = {}
            if not hasattr(g, 'trade_round_avg_cost'):
                g.trade_round_avg_cost = {}
            if not hasattr(g, 'trade_round_invest'):
                g.trade_round_invest = {}
            if not hasattr(g, 'trade_round_realized'):
                g.trade_round_realized = {}
            if not hasattr(g, 'trade_round_logs'):
                g.trade_round_logs = {}
            if not hasattr(g, 'trade_round_result'):
                g.trade_round_result = []
            # =======================================

            def _fmt2(x):
                try:
                    if x is None:
                        return 'None'
                    return '{:.2f}'.format(float(x))
                except Exception:
                    return str(x)

            # ========== 卖出全仓 ==========
            if action == 'SELL_ALL':
                if destination is None:
                    destination = 'pre'

                # 取卖出价（若sig没带，尽量现取）
                exec_price = sig.get('price', None)
                if exec_price is None:
                    exec_price = sig.get('r_p', None)
                if exec_price is None:
                    try:
                        now_dt = get_datetime()
                        today_str = now_dt.strftime('%Y-%m-%d')
                        market_open = today_str + ' 09:30:00'
                        df = DataModule.get_today_minute_data(stock, market_open, now_dt)
                        if df is not None and len(df) > 0:
                            exec_price = float(df['close'].iloc[-1])
                    except Exception:
                        exec_price = None

                # ====== 先结算本轮收益（POS->0）======
                try:
                    qty_left = float(g.trade_round_qty.get(stock, 0.0))
                    avg_cost = g.trade_round_avg_cost.get(stock, None)
                    invest = float(g.trade_round_invest.get(stock, 0.0))
                    realized = float(g.trade_round_realized.get(stock, 0.0))

                    if exec_price is not None and avg_cost is not None and qty_left > 0:
                        px = float(exec_price)
                        avg = float(avg_cost)
                        realized = realized + (px - avg) * qty_left

                        today_str = get_datetime().strftime('%Y-%m-%d')
                        if stock not in g.trade_round_logs:
                            g.trade_round_logs[stock] = []
                        g.trade_round_logs[stock].append({
                            'type': 'SELL_FINAL',
                            'date': today_str,
                            'qty': qty_left,
                            'price': px,
                            'avg_cost': avg
                        })

                    ret_pct = None
                    if invest > 0:
                        ret_pct = realized / invest * 100.0

                    log.info('【本轮交易结算】{} | 卖出触发:{} | 最终卖出价:{} | 总投入:{} | 已实现盈亏:{} | 收益率:{}%'.format(
                        stock, str(rule), _fmt2(exec_price), _fmt2(invest), _fmt2(realized), _fmt2(ret_pct)
                    ))

                    g.trade_round_result.append({
                        'stock': stock,
                        'end_date': get_datetime().strftime('%Y-%m-%d'),
                        'rule': rule,
                        'final_price': exec_price,
                        'invest': invest,
                        'realized': realized,
                        'ret_pct': ret_pct
                    })

                except Exception as e:
                    log.info('【本轮交易结算】{} 失败: {}'.format(stock, str(e)))

                # 清理本轮追踪
                try:
                    if stock in g.trade_round_qty:
                        del g.trade_round_qty[stock]
                    if stock in g.trade_round_avg_cost:
                        del g.trade_round_avg_cost[stock]
                    if stock in g.trade_round_invest:
                        del g.trade_round_invest[stock]
                    if stock in g.trade_round_realized:
                        del g.trade_round_realized[stock]
                    if stock in g.trade_round_logs:
                        del g.trade_round_logs[stock]
                except Exception:
                    pass
                # ====================================

                sell_info = dict(sig)
                sell_info['action'] = action
                sell_info['destination'] = destination
                sell_info['price'] = exec_price

                PoolManager.add_sell_all_pending(stock, sell_info)

                # 清理“半仓当日锁”
                if stock in g.pos_half_lock_date:
                    del g.pos_half_lock_date[stock]

                log.info('【执行动作】{} | action:{} | dest:{} | 触发条件:{} | price:{}'.format(
                    stock, action, destination, rule, _fmt2(exec_price)
                ))

                return True, destination, sell_info

            # ========== 卖出半仓 ==========
            if action == 'SELL_HALF':
                if not hasattr(g, 'buy_pos') or stock not in g.buy_pos:
                    return False, None, None

                # 取卖出价（若sig没带，尽量现取）
                exec_price = sig.get('price', None)
                if exec_price is None:
                    exec_price = sig.get('r_p', None)
                if exec_price is None:
                    try:
                        now_dt = get_datetime()
                        today_str = now_dt.strftime('%Y-%m-%d')
                        market_open = today_str + ' 09:30:00'
                        df = DataModule.get_today_minute_data(stock, market_open, now_dt)
                        if df is not None and len(df) > 0:
                            exec_price = float(df['close'].iloc[-1])
                    except Exception:
                        exec_price = None

                g.buy_pos[stock] = 0.5

                # 记录：当日从1→0.5后默认锁定
                today_str = get_datetime().strftime('%Y-%m-%d')
                g.pos_half_lock_date[stock] = today_str

                # 收益追踪：卖出0.5记已实现盈亏
                try:
                    sell_qty = 0.5
                    if exec_price is not None and g.trade_round_avg_cost.get(stock, None) is not None:
                        avg_cost = float(g.trade_round_avg_cost.get(stock))
                        px = float(exec_price)
                        g.trade_round_realized[stock] = float(g.trade_round_realized.get(stock, 0.0)) + (px - avg_cost) * sell_qty
                        g.trade_round_qty[stock] = float(g.trade_round_qty.get(stock, 1.0)) - sell_qty

                        if stock not in g.trade_round_logs:
                            g.trade_round_logs[stock] = []
                        g.trade_round_logs[stock].append({
                            'type': 'SELL_HALF',
                            'date': today_str,
                            'qty': sell_qty,
                            'price': px,
                            'avg_cost': avg_cost
                        })
                except Exception as e:
                    log.info('【收益追踪】{} SELL_HALF记账失败: {}'.format(stock, str(e)))

                # 关键事件打标：SELL_HALF
                if not hasattr(g, 'xh_event_flag'):
                    g.xh_event_flag = {}
                if not hasattr(g, 'xh_event_type'):
                    g.xh_event_type = {}
                g.xh_event_flag[stock] = today_str
                g.xh_event_type[stock] = 'SELL_HALF'

                info = dict(sig)
                info['action'] = action
                info['pos_after'] = 0.5
                info['price'] = exec_price

                log.info('【执行动作】{} | action:{} | 触发条件:{} | POS=>0.5 | price:{} | 事件打标:SELL_HALF'.format(
                    stock, action, rule, _fmt2(exec_price)
                ))

                return True, destination, info

            # ========== 回补 ==========
            if action == 'REBUY':
                if not hasattr(g, 'buy_pos') or stock not in g.buy_pos:
                    return False, None, None

                exec_price = sig.get('price', None)
                if exec_price is None:
                    exec_price = sig.get('r_p', None)
                if exec_price is None:
                    try:
                        now_dt = get_datetime()
                        today_str = now_dt.strftime('%Y-%m-%d')
                        market_open = today_str + ' 09:30:00'
                        df = DataModule.get_today_minute_data(stock, market_open, now_dt)
                        if df is not None and len(df) > 0:
                            exec_price = float(df['close'].iloc[-1])
                    except Exception:
                        exec_price = None

                g.buy_pos[stock] = 1.0

                # ✅ 文档要求：回补后 DAYS=3
                if not hasattr(g, 'buy_days'):
                    g.buy_days = {}
                g.buy_days[stock] = 3

                # ✅ 文档要求：记录 REBUY-DATE（回补当日标记）
                if not hasattr(g, 'rebuy_date'):
                    g.rebuy_date = {}
                g.rebuy_date[stock] = get_datetime().strftime('%Y-%m-%d')

                # ✅ 文档要求：回补后清除 PBL（回补线不应残留）
                try:
                    if hasattr(g, 'sell_pbl'):
                        g.sell_pbl[stock] = None
                    if hasattr(g, 'sell_pbl_t'):
                        g.sell_pbl_t[stock] = None
                    if hasattr(g, 'sell_pbl_src'):
                        g.sell_pbl_src[stock] = None
                except Exception:
                    pass

                # 回补后解除“半仓当日锁”
                if stock in g.pos_half_lock_date:
                    del g.pos_half_lock_date[stock]

                # 收益追踪：回补0.5更新投入与加权成本
                try:
                    buy_qty = 0.5
                    if exec_price is not None:
                        px = float(exec_price)

                        old_qty = float(g.trade_round_qty.get(stock, 0.5))
                        old_avg = g.trade_round_avg_cost.get(stock, None)
                        old_invest = float(g.trade_round_invest.get(stock, 0.0))

                        add_invest = buy_qty * px
                        new_invest = old_invest + add_invest
                        g.trade_round_invest[stock] = new_invest

                        if old_avg is None:
                            new_avg = px
                        else:
                            old_avg = float(old_avg)
                            new_avg = ((old_qty * old_avg) + (buy_qty * px)) / (old_qty + buy_qty)

                        g.trade_round_avg_cost[stock] = new_avg
                        g.trade_round_qty[stock] = old_qty + buy_qty

                        today_str = get_datetime().strftime('%Y-%m-%d')
                        if stock not in g.trade_round_logs:
                            g.trade_round_logs[stock] = []
                        g.trade_round_logs[stock].append({
                            'type': 'REBUY',
                            'date': today_str,
                            'qty': buy_qty,
                            'price': px,
                            'new_avg_cost': new_avg
                        })

                except Exception as e:
                    log.info('【收益追踪】{} REBUY记账失败: {}'.format(stock, str(e)))

                # 关键事件打标：REBUY
                today_str = get_datetime().strftime('%Y-%m-%d')
                if not hasattr(g, 'xh_event_flag'):
                    g.xh_event_flag = {}
                if not hasattr(g, 'xh_event_type'):
                    g.xh_event_type = {}
                g.xh_event_flag[stock] = today_str
                g.xh_event_type[stock] = 'REBUY'

                info = dict(sig)
                info['action'] = action
                info['pos_after'] = 1.0
                info['price'] = exec_price

                log.info('【执行动作】{} | action:{} | 触发条件:{} | POS=>1.0 | price:{} | 事件打标:REBUY'.format(
                    stock, action, rule, _fmt2(exec_price)
                ))

                return True, destination, info

            return False, None, None

        except Exception as e:
            log.error('执行卖出动作失败 {}: {}'.format(stock, str(e)))
            return False, None, None


    @staticmethod
    def reset_daily_sell_tracking():
        """
        重置当日卖出/回补跟踪变量
        按02文档：每日盘前清空 PSL/PSL-T/PBL/TRG-TIME
        以及：尾盘兜底执行标记（sell_1455_done）

        补丁：引入
        - PSL-T 支持 'empty'
        - PBL-T 支持 'full'/'half'（为回补半仓做准备）
        """
        g.sell_psl = {}
        g.sell_psl_t = {}      # 允许写入 'full'/'half'/'empty'/None
        g.sell_pbl = {}
        g.sell_pbl_t = {}      # 'full'/'half'/None（后续回补用）
        g.sell_trg_time = {}
        g.sell_1455_done = {}

    @staticmethod
    def record_yesterday_macd(today_str):
        """
        记录买入仓所有股票的昨日MACD柱状值（T-1-MACD）
        每日盘后调用
        """
        success_count = 0
        for stock in g.buy_pool:
            macd_hist = DataModule.calculate_macd_hist(stock, today_str)
            if macd_hist is not None:
                g.yesterday_macd[stock] = macd_hist
                success_count += 1
        return success_count

    @staticmethod
    def increase_buy_days_after_close(today_str):
        """
        盘后把买入仓持仓天数 +1
        约定：买入当日 DAYS=1；每个交易日盘后递增一次
        """
        if not hasattr(g, 'buy_days'):
            return 0
        if not hasattr(g, 'buy_pool'):
            return 0

        updated = 0
        for stock in g.buy_pool:
            old = g.buy_days.get(stock, 0)
            try:
                old = int(old)
            except Exception:
                old = 0
            g.buy_days[stock] = old + 1
            updated += 1

        if updated > 0:
            log.info('【盘后】买入仓DAYS递增: {} 只 | 样本(前5): {}'.format(
                updated,
                [(s, g.buy_days.get(s)) for s in g.buy_pool[:5]]
            ))
        return updated
        
    @staticmethod
    def refresh_t1_cache_for_pools():
        """
        每日盘前统一刷新所有 T-1-X（检测当日的前一交易日口径）
        覆盖：T-1-O/H/L/C/DIF/MACD

        同时做兼容写入：
        - g.buy_t1_h / g.buy_t1_dif / g.yesterday_macd
        """
        now_dt = get_datetime()
        ydt = get_previous_trading_date(now_dt)
        ystr = ydt.strftime('%Y-%m-%d')

        # 若当天已刷新过，避免重复计算
        if hasattr(g, 't1_cache_date') and g.t1_cache_date == ystr:
            return 0

        if not hasattr(g, 't1_cache'):
            g.t1_cache = {}
        g.t1_cache_date = ystr

        # 至少覆盖 buy_pool；建议同时覆盖 ready_pool（如果你的买入/卖出规则也用到T-1）
        universe = []
        if hasattr(g, 'buy_pool') and g.buy_pool:
            universe.extend(list(g.buy_pool))
        if hasattr(g, 'ready_pool') and g.ready_pool:
            universe.extend(list(g.ready_pool))
        universe = list(set(universe))

        # 兼容容器
        if not hasattr(g, 'buy_t1_h'):
            g.buy_t1_h = {}
        if not hasattr(g, 'buy_t1_dif'):
            g.buy_t1_dif = {}
        if not hasattr(g, 'yesterday_macd'):
            g.yesterday_macd = {}

        updated = 0

        for stock in universe:
            t1_o = t1_h = t1_l = t1_c = None
            t1_dif = t1_macd = None

            # ===== 取昨日日线 O/H/L/C（严格按昨日口径）=====
            try:
                df = DataModule.get_daily_data(stock, ystr, 5)
                if df is not None and len(df) > 0:
                    # 过滤非真实交易日/停牌等（与你MA过滤口径一致）
                    if 'volume' in df.columns:
                        df = df[df['volume'] > 100]
                    if len(df) > 0:
                        row = df.iloc[-1]
                        t1_o = float(row['open']) if 'open' in row else None
                        t1_h = float(row['high']) if 'high' in row else None
                        t1_l = float(row['low']) if 'low' in row else None
                        t1_c = float(row['close']) if 'close' in row else None
            except Exception as e:
                log.info('【T-1刷新】{} 取昨日日线失败: {}'.format(stock, str(e)))

            # ===== 取昨日 DIF / MACD（盘后指标，昨日口径）=====
            try:
                t1_dif = DataModule.calculate_macd_dif(stock, ystr)
                t1_dif = float(t1_dif) if t1_dif is not None else None
            except Exception as e:
                log.info('【T-1刷新】{} 取T-1-DIF失败: {}'.format(stock, str(e)))

            try:
                t1_macd = DataModule.calculate_macd_hist(stock, ystr)
                t1_macd = float(t1_macd) if t1_macd is not None else None
            except Exception as e:
                log.info('【T-1刷新】{} 取T-1-MACD失败: {}'.format(stock, str(e)))

            g.t1_cache[stock] = {
                'date': ystr,
                'o': t1_o, 'h': t1_h, 'l': t1_l, 'c': t1_c,
                'dif': t1_dif,
                'macd': t1_macd
            }

            # ===== 兼容写入（你现有代码还在用这些字段）=====
            if t1_h is not None:
                g.buy_t1_h[stock] = t1_h
            if t1_dif is not None:
                g.buy_t1_dif[stock] = t1_dif
            if t1_macd is not None:
                g.yesterday_macd[stock] = t1_macd

            updated += 1

        log.info('【盘前统一刷新T-1缓存】昨日:{} | 覆盖:{}只 | 样本(前3):{}'.format(
            ystr, updated,
            [(s,
              g.t1_cache.get(s, {}).get('h'),
              g.t1_cache.get(s, {}).get('dif'),
              g.t1_cache.get(s, {}).get('macd')) for s in universe[:3]]
        ))
        return updated

    @staticmethod
    def _cleanup_stock_on_exit(stock):
        """
        STEP7：统一“彻底离场”清理口径，防止幽灵状态。
        约定：只要最终去向为 exit，就必须调用本方法。
        """
        # ---------- 1) 从各池移除 ----------
        try:
            if hasattr(g, 'pre_pool') and stock in g.pre_pool:
                g.pre_pool.remove(stock)
        except Exception:
            pass

        try:
            if hasattr(g, 'ready_pool') and stock in g.ready_pool:
                g.ready_pool.remove(stock)
        except Exception:
            pass

        try:
            if hasattr(g, 'buy_pool') and stock in g.buy_pool:
                g.buy_pool.remove(stock)
        except Exception:
            pass

        # ---------- 2) 统一清理 dict 容器 ----------
        # 原则：exit 就清掉所有追踪痕迹（包含买入/卖出线、打标、缓存、收益追踪等）
        dict_attrs = [
            # 预备/准备阶段追踪
            'entry_date', 'decline_count', 'prev_high', 'prev_dif', 'transfer_info',
            'v55m', 'v55m_date',
            'yesterday_dif', 'dif_trigger_time', 'buy_line', 'buy_line_t',
            'buy_forbidden_today', 'half_buy_pending_return',

            # 买入仓追踪
            'buy_info', 'buy_pos', 'buy_days', 'buy_bp', 'buy_time',
            'buy_pbl', 'buy_pbl_t',
            'buy_xh', 'buy_xh233', 'buy_xh233_date',
            'buy_t1_h', 'buy_t1_dif', 'buy_dc',

            # 卖出系统追踪
            'yesterday_macd',
            'sell_psl', 'sell_psl_t', 'sell_psl_src',
            'sell_pbl', 'sell_pbl_t', 'sell_pbl_src',
            'sell_trg_time', 'sell_1455_done',
            'rebuy_date',
            'pos_half_lock_date', 'half_lock_l1c_allow_date', 'half_lock_l1c_allow_time',

            # 锚点事件打标
            'xh_event_flag', 'xh_event_type', 'xh_event_date',

            # T-1缓存
            't1_cache',

            # SELL_ALL pending 容器（如果仍存在）
            'sell_all_pending',
        ]

        for attr in dict_attrs:
            try:
                if hasattr(g, attr):
                    d = getattr(g, attr)
                    if isinstance(d, dict) and stock in d:
                        del d[stock]
            except Exception:
                pass

        # ---------- 3) 收益追踪清理（可选但建议 exit 一并清） ----------
        for attr in ['trade_round_qty', 'trade_round_avg_cost', 'trade_round_invest', 'trade_round_realized', 'trade_round_logs']:
            try:
                if hasattr(g, attr):
                    d = getattr(g, attr)
                    if isinstance(d, dict) and stock in d:
                        del d[stock]
            except Exception:
                pass

    
# ==================== 7. 交易执行类（实盘用）====================
class TradeExecutor:
    """实盘交易执行和资金管理"""
    
    @staticmethod
    def get_available_cash():
        """
        获取可用资金
        返回: float
        """
        try:
            return context.portfolio.cash
        except:
            return 0
    
    @staticmethod
    def calculate_position_size(stock, target_percent):
        """
        计算买入数量
        
        返回: int (股数)
        """
        try:
            # TODO: 实现仓位计算逻辑
            pass
        except Exception as e:
            log.error('{} 仓位计算失败: {}'.format(stock, str(e)))
            return 0
    
    @staticmethod
    def execute_real_buy(stock, amount):
        """
        执行真实买入
        资金充裕时调用
        
        返回: bool (是否成功)
        """
        try:
            # TODO: 实现真实买入逻辑
            # order_target_percent() 或 order()
            pass
        except Exception as e:
            log.error('{} 真实买入失败: {}'.format(stock, str(e)))
            return False
    
    @staticmethod
    def execute_virtual_buy(stock, price):
        """
        执行虚拟买入
        资金不足时调用，记录到买入仓
        
        返回: bool
        """
        try:
            # TODO: 记录虚拟买入信息
            pass
        except Exception as e:
            log.error('{} 虚拟买入失败: {}'.format(stock, str(e)))
            return False
    
    @staticmethod
    def execute_sell(stock):
        """
        执行卖出
        
        返回: bool
        """
        try:
            # TODO: 实现卖出逻辑
            # order_target(stock, 0)
            pass
        except Exception as e:
            log.error('{} 卖出失败: {}'.format(stock, str(e)))
            return False
    
    @staticmethod
    def check_and_execute_buy_from_ready():
        """
        检查资金并从准备仓执行买入
        资金充裕: 真实买入进持仓
        资金不足: 虚拟买入进买入仓
        """
        try:
            # TODO: 实现资金检查和买入执行逻辑
            pass
        except Exception as e:
            log.error('从准备仓买入执行失败: {}'.format(str(e)))

# ==================== 8. 编排器类（实盘用）====================
class MainOrchestrator(object):
    """
    主流程编排器：主程序入口函数(handle_bar/after_trading)只调用这里。
    这里负责调度各阶段扫描与兜底，不改变任何交易逻辑，仅做结构下沉。
    """
    @staticmethod
    def _run_after_trading_pipeline():
        """
        盘后流水线（核心）：
        1) 盘后选股（创新高）→ 加入预备仓
        2) 更新预备仓/准备仓累计下降次数（DC）
        2.5) 增量更新 V55M（成交量55日新高标记）
        3) 批量检查准备仓是否出仓（如有）
        4) 记录准备仓昨日DIF（T-1-DIF，供次日买入触发用）
        """
        try:
            today_str = get_datetime().strftime('%Y-%m-%d')

            log.info('=' * 70)
            log.info('【{}盘后】'.format(today_str))

            # 1) 盘后选股 → 预备仓
            selected = StockSelector.select_new_high_stocks(today_str)
            added_count, newly_added = PoolManager.add_to_pre_pool(selected)
            log.info('盘后选股: {} 只 | 新增入预备仓: {} 只'.format(len(selected), added_count))
            if newly_added:
                log.info('新增预备仓样本(前10): {}'.format(newly_added[:10]))

            # 2) 更新累计下降次数（DC）
            decline_summary = PoolManager.update_all_decline_count(today_str)
            log.info('累计下降更新: 覆盖 {} 只'.format(len(decline_summary)))

            # 2.5) 增量更新 V55M（方案1）
            v55m_updated = PoolManager.update_v55m_flags(today_str)
            log.info('V55M增量更新: 新增True {} 只'.format(v55m_updated))

            # 3) 准备仓出仓检查
            exit_list = PoolManager.process_ready_pool_exit(today_str)
            if exit_list:
                log.info('准备仓盘后出仓: {} 只 | 样本: {}'.format(len(exit_list), exit_list[:10]))
            else:
                log.info('准备仓盘后出仓: 0 只')

            # 4) 记录准备仓昨日DIF（用于次日 check_ready_to_buy 的 T-1-DIF）
            dif_count = PoolManager.record_yesterday_dif(today_str)
            log.info('记录准备仓昨日DIF: {} 只'.format(dif_count))

            # 盘后仓位概览
            status = PoolManager.get_pool_status()
            log.info('盘后仓位: 预备仓:{} | 准备仓:{} | 买入仓:{}'.format(
                status['pre_count'], status['ready_count'], status['buy_count']
            ))
            log.info('=' * 70)

        except Exception as e:
            log.error('盘后流水线执行失败: {}'.format(str(e)))

    @staticmethod
    def on_bar(context, bar_dict):
        MainOrchestrator._run_pre_to_ready()
        MainOrchestrator._run_ready_to_buy()
        MainOrchestrator._run_ready_to_buy_failsafe_1455()
        MainOrchestrator._run_buy_to_sell()
        # 若你后续要加实盘执行，也放在这里统一调度
        # MainOrchestrator._run_live_execution(context, bar_dict)

    @staticmethod
    def on_after_trading(context):
        today_str = get_datetime().strftime('%Y-%m-%d')

        MainOrchestrator._run_after_trading_pipeline()

        # ——盘后处理“半仓路径未买入”的回退——
        MainOrchestrator._run_ready_half_pending_return_after_close(today_str)

        PoolManager.record_yesterday_macd(today_str)

        PoolManager.increase_buy_days_after_close(today_str)

        # 盘后EOD卖出（可能触发SELL_ALL，STEP2里会先登记到sell_all_pending）
        MainOrchestrator._run_buy_to_sell_eod()

        # ✅ STEP2：对当日所有 SELL_ALL 统一进行“盘后去向判定”
        PoolManager.process_sell_all_pending_after_close(today_str)

        MainOrchestrator._update_buy_pool_xh_after_close()

        MainOrchestrator._reset_intraday_flags()


    # =========================
    # 盘中：各阶段扫描
    # =========================
    @staticmethod
    def _run_pre_to_ready():
        for stock in g.pre_pool[:]:
            is_triggered, info = TriggerChecker.check_pre_to_ready(stock)
            if is_triggered:
                PoolManager.transfer_pre_to_ready(stock, info)
                if hasattr(g, 'today_transferred'):
                    g.today_transferred.append(stock)

    @staticmethod
    def _run_ready_to_buy():
        for stock in g.ready_pool[:]:
            is_buy, buy_info = TriggerChecker.check_ready_to_buy(stock)
            if is_buy:
                PoolManager.transfer_ready_to_buy(stock, buy_info)

    @staticmethod
    def _run_buy_to_sell():
        """
        盘中：买入仓卖出检测（统一走 SellRuleEngine + apply_action_from_sell_signal）
        目的：
        1) SELL_ALL 必须登记到 pending，盘后统一分流去向（避免盘中直接转仓污染）
        2) 所有 POS 变化都由 apply_action_from_sell_signal 统一结算与清理
        """
        try:
            for stock in g.buy_pool[:]:
                sig = SellRuleEngine.check_intraday(stock)
                if sig is None:
                    continue
                PoolManager.apply_action_from_sell_signal(stock, sig)
        except Exception as e:
            log.error('盘中买入仓卖出检测失败: {}'.format(str(e)))

    @staticmethod
    def _run_buy_to_sell_eod():
        """
        盘后评估买入仓卖出规则（EOD）
        """
        try:
            today_str = get_datetime().strftime('%Y-%m-%d')

            for stock in g.buy_pool[:]:
                sig = SellRuleEngine.check_eod(stock, today_str)
                if sig is None:
                    continue
                PoolManager.apply_action_from_sell_signal(stock, sig)

        except Exception as e:
            log.error('盘后卖出规则执行失败: {}'.format(str(e)))

    # =========================
    # 盘中：14:55 半仓兜底（从handle_bar下沉）
    # =========================
    
    @staticmethod
    def _run_ready_to_buy_failsafe_1455():
        current_time = get_datetime()
        today_str = current_time.strftime('%Y-%m-%d')
        hm = current_time.strftime('%H:%M')
        if hm != '14:55':
            return

        if not hasattr(g, 'buy_line'):
            return
        if not hasattr(g, 'buy_line_t'):
            g.buy_line_t = {}
        if not hasattr(g, 'half_buy_pending_return'):
            g.half_buy_pending_return = {}

        # 确保“当日禁触发”字典存在
        if not hasattr(g, 'buy_forbidden_today'):
            g.buy_forbidden_today = {}

        for stock in g.ready_pool[:]:
            if stock not in g.buy_line:
                continue

            buy_line_price = g.buy_line.get(stock, None)
            if buy_line_price is None:
                continue

            try:
                # ✅ H-233 口径：不含今日
                yesterday_dt = get_previous_trading_date(current_time)
                high_233 = DataModule.get_highest_price(stock, StrategyConfig.LOOKBACK_DAYS, yesterday_dt)
                if high_233 is None:
                    continue

                # 半仓路径判定：PBL >= H-233 * 0.97
                if float(buy_line_price) >= float(high_233) * 0.97:
                    # 14:55：不立刻转仓，改为“盘后转回预备仓”
                    g.half_buy_pending_return[stock] = today_str

                    # 清理TRG-TIME与买入线
                    if hasattr(g, 'dif_trigger_time') and stock in g.dif_trigger_time:
                        del g.dif_trigger_time[stock]
                    if hasattr(g, 'buy_line') and stock in g.buy_line:
                        del g.buy_line[stock]
                    if hasattr(g, 'buy_line_t') and stock in g.buy_line_t:
                        del g.buy_line_t[stock]

                    # 当日不再触发任何操作
                    g.buy_forbidden_today[stock] = today_str

                    log.info('【14:55兜底-半仓未买】{} | PBL:{:.2f} | H-233:{:.2f} | 阈值:{:.2f} => 标记盘后转回预备仓 + 清理TRG/PBL + 当日禁触发'.format(
                        stock, float(buy_line_price), float(high_233), float(high_233) * 0.97
                    ))

            except Exception as e:
                log.info('【14:55兜底】{} 处理异常: {}'.format(stock, str(e)))

    # =========================
    # 盘后：示例（X-H补齐 + 日内标记重置）
    # 这里我先给“空实现/保守实现”，避免你当前阶段误改逻辑
    # =========================
    @staticmethod
    def _update_buy_pool_xh_after_close():
        """
        盘后统一回填事件锚点：
        - X-H：事件日盘后最高（取事件日的日线high）
        - XH-233：事件日前233日最高（不含事件日；截止事件日前一交易日）
        - XH-233_DATE：XH-233 最高价发生日
        并将 g.xh_event_date[stock] 落地为“事件日”
        """
        if not hasattr(g, 'xh_event_flag') or not hasattr(g, 'xh_event_type'):
            return
        if not hasattr(g, 'buy_pool') or len(g.buy_pool) == 0:
            return

        now_dt = get_datetime()
        today_str = now_dt.strftime('%Y-%m-%d')

        if not hasattr(g, 'buy_xh'):
            g.buy_xh = {}
        if not hasattr(g, 'buy_xh233'):
            g.buy_xh233 = {}
        if not hasattr(g, 'buy_xh233_date'):
            g.buy_xh233_date = {}
        if not hasattr(g, 'xh_event_date'):
            g.xh_event_date = {}

        for stock in list(g.buy_pool):
            # 仅处理当日被打标的股票
            if g.xh_event_flag.get(stock, None) != today_str:
                continue

            event_type = g.xh_event_type.get(stock, 'UNKNOWN')

            # ===== 1) X-H：事件日盘后最高（事件日=今天；取今天日线high）=====
            x_h = None
            try:
                df_today = DataModule.get_daily_data(stock, today_str, 1)
                if df_today is not None and len(df_today) > 0:
                    x_h = float(df_today['high'].iloc[-1])
            except Exception as e:
                log.info('【锚点回填】{} 获取X-H失败: {}'.format(stock, str(e)))

            # ===== 2) XH-233 / XH-233_DATE：事件日前233日最高（截止“昨天”）=====
            xh_233 = None
            xh_233_date = None
            try:
                yesterday_dt = get_previous_trading_date(now_dt)
                xh_233, xh_233_date = DataModule.get_highest_price_with_date(
                    stock, StrategyConfig.LOOKBACK_DAYS, yesterday_dt
                )
            except Exception as e:
                log.info('【锚点回填】{} 获取XH-233失败: {}'.format(stock, str(e)))

            # 写回
            g.buy_xh[stock] = x_h
            g.buy_xh233[stock] = xh_233
            g.buy_xh233_date[stock] = xh_233_date

            # 事件日落地
            g.xh_event_date[stock] = today_str

            # 清除待处理标记（避免重复）
            del g.xh_event_flag[stock]

            log.info('【锚点回填】{} | 事件:{} | 事件日:{} | X-H:{} | XH-233:{} | XH-233_DATE:{}'.format(
                stock, event_type, today_str, str(x_h), str(xh_233), str(xh_233_date)
            ))

    @staticmethod
    def _reset_intraday_flags():
        # 例如：当日禁买标记 buy_forbidden_today 可以保留按日期字典，不必清空
        # 若你有 today_transferred / 当日临时缓存，建议在这里统一清空
        if hasattr(g, 'today_transferred'):
            g.today_transferred = []

    @staticmethod
    def _run_ready_half_pending_return_after_close(today_str):
        """
        盘后执行：半仓路径在14:55仍未触发买入 → 转回预备仓
        同时：该股票当日已被 buy_forbidden_today 标记，不会再触发
        """
        if not hasattr(g, 'half_buy_pending_return'):
            return
        if not hasattr(g, 'ready_pool') or not hasattr(g, 'pre_pool'):
            return

        pending = [s for s, d in g.half_buy_pending_return.items() if d == today_str]
        if not pending:
            return

        moved = []
        for stock in pending:
            if stock in g.ready_pool:
                g.ready_pool.remove(stock)
                if stock not in g.pre_pool:
                    g.pre_pool.append(stock)
                moved.append(stock)

        # 清理当天标记（只清当日）
        for stock in moved:
            try:
                del g.half_buy_pending_return[stock]
            except Exception:
                pass

        if moved:
            log.info('【盘后回退-半仓未买】日期:{} | 回退到预备仓:{}只 | 样本(前10):{}'.format(
                today_str, len(moved), moved[:10]
            ))

# ==================== 策略主程序 ====================

def init(context):
    """初始化"""
    set_benchmark('399006.SZ')
    set_commission(PerShare(type='stock', cost=0.0002))
    set_slippage(PriceSlippage(0.005))

    # ========= 兼容补丁：ShowLog 无 warning 方法，避免回测中断 =========
    try:
        if not hasattr(log, 'warning'):
            log.info = log.info
    except Exception:
        pass
    # ===============================================================

    # 初始化各个仓位
    g.pre_pool = []           # 预备仓
    g.ready_pool = []         # 准备仓
    g.buy_pool = []           # 买入仓

    # 初始化信息记录
    g.transfer_info = {}      # 转仓记录（累积）
    g.buy_info = {}           # 买入记录（包含POS/DAYS/X-H等）
    g.sell_info = {}          # 卖出记录
    g.exit_info = {}          # 出仓记录

    g.today_transferred = []  # 当日转入准备仓的股票列表

    # 出仓逻辑相关变量
    g.entry_date = {}         # 记录股票进入预备仓的日期
    g.decline_count = {}      # 记录下降次数
    g.prev_high = {}          # 记录前一日最高价
    g.prev_dif = {}           # 记录前一日DIF值

    # ========== V55M 动态标记（新增）==========
    g.v55m = {}               # {stock: True/False}
    g.v55m_date = {}          # {stock: 'YYYY-MM-DD'} 第一次变True的日期
    # ========================================

    # ========== 买入逻辑相关变量 ==========
    g.yesterday_dif = {}      # 记录昨日盘后DIF值
    g.dif_trigger_time = {}   # 记录DIF触发时刻（datetime对象）
    g.buy_line = {}           # 记录预买入线价格（PBL）
    g.buy_line_t = {}         # 记录预买入线类型（PBL-T：'full'/'half'，用于兜底识别）
    # =====================================

    # ========== 卖出/回补逻辑相关变量 ==========
    g.yesterday_macd = {}     # 记录昨日盘后MACD柱状值（T-1-MACD）

    g.sell_psl = {}           # {stock: PSL}
    g.sell_psl_t = {}         # {stock: 'full'/'half'/'empty'/None}
    g.sell_pbl = {}           # {stock: PBL}  （后续用于回补）
    g.sell_pbl_t = {}         # {stock: 'full'/'half'/None}  （后续用于回补）
    g.rebuy_date = {}         # {stock: 'YYYY-MM-DD'} 回补发生日期（REBUY-DATE）
    g.sell_trg_time = {}      # {stock: 'HH:MM'} 当日已做过整点PSL设置的时刻，避免重复
    g.t1_cache = {}           # {stock: {'date': 'YYYY-MM-DD', 'o':. 'h':. 'l':. 'c':. 'dif':. 'macd':.}}
    g.t1_cache_date = None    # 'YYYY-MM-DD' 本次缓存对应的“昨日日期”
    # =======================================

    # ========== 当日禁触发 / 半仓兜底盘后回退 ==========
    g.buy_forbidden_today = {}        # {stock: 'YYYY-MM-DD'}
    g.half_buy_pending_return = {}    # {stock: 'YYYY-MM-DD'} 14:55标记，盘后转回预备仓
    # ===============================================

    # ========== STEP2 新增：SELL_ALL 盘后去向临时列表 ==========
    g.sell_all_pending = {}   # {stock: sell_info} 盘中/盘后SELL_ALL先登记，盘后统一判定去向
    # ======================================================

    g.enable_tmp_sell_rules = False  # 默认关闭：避免干扰文档卖出回测

    log.info('策略初始化完成')


def before_trading(context):
    """盘前运行"""
    date = get_datetime().strftime('%Y-%m-%d')
    status = PoolManager.get_pool_status()

    g.today_transferred = []
    PoolManager.reset_daily_buy_tracking()
    PoolManager.reset_daily_sell_tracking()  # ← 新增：每天清空PSL/PBL/TRG-TIME
    PoolManager.refresh_t1_cache_for_pools()


    log.info('=' * 70)
    log.info('【{}开盘前】'.format(date))
    log.info('预备仓: {} 只 | 准备仓: {} 只 | 买入仓: {} 只'.format(
        status['pre_count'], status['ready_count'], status['buy_count']))
    log.info('=' * 70)


def handle_bar(context, bar_dict):
    MainOrchestrator.on_bar(context, bar_dict)


def after_trading(context):
    MainOrchestrator.on_after_trading(context)
