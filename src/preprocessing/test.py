class DataPreprocesser:
    def __init__(self):
        pass

    def __fe(self):
        pass

    def transform(self, input_data):
        # NOTE define data preprocess

        # NOTE 防止鏈式索引所引發的問題
        data = input_data.copy()

        target_encoder = TargetEncoder()
        target_encoding_colnames = []

        # data.to_frame()
        #         
        # ** (1) acqic 收單行代碼
        """
            * target encoding
        """
        target_encoding_colnames.append('acqic')
        

        # ** (2) bacno 歸戶帳號
        """
            * target encoding
            * train/test 交集的類別 
        
        """
        target_encoding_colnames.append('bacno')


        # ** (3) cano 交易卡號
        """
            * target encoding
        """
        target_encoding_colnames.append('cano')


        # ** (4) conam 交易金額-台幣(經過轉換)
        """
            * bins
        """
        
        # ** (5) contp 交易類別
        """
            * target encoding
        """
        target_encoding_colnames.append('contp')


        # ** (6) csmcu 消費地幣別
        """
            * target encoding
        """
        target_encoding_colnames.append('csmcu')


        # ** (7) ecfg 網路交易註記
        """
            * label encoding
        """
        encoded_ecfg = data.ecfg.apply(utils.binary_lebel_encoding).values
        data['encoded_ecfg'] = encoded_ecfg
        data.drop('ecfg', axis=1, inplace=True)


        # ** (8) etymd 交易型態
        """
            * target encoding
        """
        target_encoding_colnames.append('etymd')


        # ** (9) flbmk Fallback 註記
        """
            * label encoding
            * fill NA as a category
        """
        encoded_flbmk = data.flbmk.apply(utils.binary_lebel_encoding).values
        data['encoded_flbmk'] = encoded_flbmk
        data.drop('flbmk', axis=1, inplace=True)


        # ** (10) flg_3dsmk 3DS 交易註記
        """
            * label encoding
            * fill NA as a category
        """
        encoded_flg_3dsmk = data.flg_3dsmk.apply(utils.binary_lebel_encoding).values
        data['encoded_flg_3dsmk'] = encoded_flg_3dsmk
        data.drop('flg_3dsmk', axis=1, inplace=True)

    
        # ** (11) hcefg 支付形態
        """
            * target encoding
        """
        target_encoding_colnames.append('hcefg')


        # ** (12) insfg 分期交易註記
        """
            * label encoding
        """
        encoded_insfg = data.insfg.apply(utils.binary_lebel_encoding).values
        data['encoded_insfg'] = encoded_insfg
        data.drop('insfg', axis=1, inplace=True)
        

        # ** (13) iterm 分期期數
        target_encoding_colnames.append('iterm')


        # ** (14) locdt 授權日期
        data.drop('locdt', axis=1, inplace=True)


        # ** (15) loctm 授權時間
        """
            * 轉換後標準化 >> 0~24
        """
        converted_toctm = data.loctm.apply(utils.convert_time).values
        data['converted_toctm'] = converted_toctm
        data.drop('loctm', axis=1, inplace=True)


        # ** (16) mcc MCC_CODE
        # 
        target_encoding_colnames.append('mcc')


        # ** (17) mchno 特店代號
        """
            * target encoding
            * train、test交集的類別
        """
        target_encoding_colnames.append('mchno')


        # ** (18) ovrlt 超額註記碼
        """
            * label encoding        """
        encoded_ovrlt = data.ovrlt.apply(utils.binary_lebel_encoding).values
        data['encoded_ovrlt'] = encoded_ovrlt
        data.drop('ovrlt', axis=1, inplace=True)


        # ** (19) scity 消費城市
        """
            * target encoding
            * train、test交集的類別
        """
        target_encoding_colnames.append('scity')


        # ** (20) stocn 消費地國別
        target_encoding_colnames.append('stocn')


        # ** (21) stscd 狀態碼
        target_encoding_colnames.append('stscd')


        # print('target_encoding_colnames : ', target_encoding_colnames)

        # data = utils.process_target_encoding_train(data, target_encoder, target_encoding_colnames)

        return data