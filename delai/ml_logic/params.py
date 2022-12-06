import os
import numpy as np

DATASET_SIZE = os.environ.get("DATASET_SIZE")
VALIDATION_DATASET_SIZE = os.environ.get("VALIDATION_DATASET_SIZE")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
LOCAL_DATA_PATH = os.path.expanduser(os.environ.get("LOCAL_DATA_PATH"))
LOCAL_REGISTRY_PATH = os.path.expanduser(os.environ.get("LOCAL_REGISTRY_PATH"))
PROJECT = os.environ.get("PROJECT")
DATASET = os.environ.get("DATASET")

COLUMN_NAMES_RAW = ["Unnamed: 0"
,"FlightDate"
,"Year"
,"Quarter"
,"Month"
,"DayofMonth"
,"DayOfWeek"
,"Airline"
,"Operating_Airline"
,"Marketing_Airline_Network"
,"Flight_Number_Marketing_Airline"
,"Origin"
,"Dest"
,"CRSDepTime"
,"OriginAirportID"
,"OriginCityName"
,"OriginStateName"
,"DestAirportID"
,"DestCityName"
,"DestStateName"
,"CRSArrTime"
,"Distance"
,"ArrDelayMinutes"
,"Cancelled"
,"Diverted"]

COLUMN_NAMES_PROCESSED = ['y_2018', 'y_2019','y_2020', 'y_2021', 'y_2022',
'dist_scaled',
'sin_dow', 'cos_dow',
'sin_dom', 'cos_dom',
'sin_month', 'cos_month',
'sin_qua', 'cos_qua',
'sin_dep', 'cos_dep',
'sin_arr', 'cos_arr',
'Marketing_Airline_Network', 'Origin', 'Dest'
,"y"
,"o_ABE"
,"o_ABI"
,"o_ABQ"
,"o_ABR"
,"o_ABY"
,"o_ACK"
,"o_ACT"
,"o_ACV"
,"o_ACY"
,"o_ADK"
,"o_ADQ"
,"o_AEX"
,"o_AGS"
,"o_AKN"
,"o_ALB"
,"o_ALO"
,"o_ALS"
,"o_ALW"
,"o_AMA"
,"o_ANC"
,"o_APN"
,"o_ART"
,"o_ASE"
,"o_ATL"
,"o_ATW"
,"o_ATY"
,"o_AUS"
,"o_AVL"
,"o_AVP"
,"o_AZA"
,"o_AZO"
,"o_BDL"
,"o_BET"
,"o_BFF"
,"o_BFL"
,"o_BFM"
,"o_BGM"
,"o_BGR"
,"o_BHM"
,"o_BIH"
,"o_BIL"
,"o_BIS"
,"o_BJI"
,"o_BKG"
,"o_BLI"
,"o_BLV"
,"o_BMI"
,"o_BNA"
,"o_BOI"
,"o_BOS"
,"o_BPT"
,"o_BQK"
,"o_BQN"
,"o_BRD"
,"o_BRO"
,"o_BRW"
,"o_BTM"
,"o_BTR"
,"o_BTV"
,"o_BUF"
,"o_BUR"
,"o_BWI"
,"o_BZN"
,"o_CAE"
,"o_CAK"
,"o_CDB"
,"o_CDC"
,"o_CDV"
,"o_CGI"
,"o_CHA"
,"o_CHO"
,"o_CHS"
,"o_CID"
,"o_CIU"
,"o_CKB"
,"o_CLE"
,"o_CLL"
,"o_CLT"
,"o_CMH"
,"o_CMI"
,"o_CMX"
,"o_CNY"
,"o_COD"
,"o_COS"
,"o_COU"
,"o_CPR"
,"o_CRP"
,"o_CRW"
,"o_CSG"
,"o_CVG"
,"o_CWA"
,"o_CYS"
,"o_DAB"
,"o_DAL"
,"o_DAY"
,"o_DBQ"
,"o_DCA"
,"o_DDC"
,"o_DEC"
,"o_DEN"
,"o_DFW"
,"o_DHN"
,"o_DIK"
,"o_DLG"
,"o_DLH"
,"o_DRO"
,"o_DRT"
,"o_DSM"
,"o_DTW"
,"o_DUT"
,"o_DVL"
,"o_EAR"
,"o_EAT"
,"o_EAU"
,"o_ECP"
,"o_EGE"
,"o_EKO"
,"o_ELM"
,"o_ELP"
,"o_ERI"
,"o_ESC"
,"o_EUG"
,"o_EVV"
,"o_EWN"
,"o_EWR"
,"o_EYW"
,"o_FAI"
,"o_FAR"
,"o_FAT"
,"o_FAY"
,"o_FCA"
,"o_FLG"
,"o_FLL"
,"o_FLO"
,"o_FNT"
,"o_FOD"
,"o_FSD"
,"o_FSM"
,"o_FWA"
,"o_GCC"
,"o_GCK"
,"o_GEG"
,"o_GFK"
,"o_GGG"
,"o_GJT"
,"o_GNV"
,"o_GPT"
,"o_GRB"
,"o_GRI"
,"o_GRK"
,"o_GRR"
,"o_GSO"
,"o_GSP"
,"o_GST"
,"o_GTF"
,"o_GTR"
,"o_GUC"
,"o_GUM"
,"o_HDN"
,"o_HGR"
,"o_HHH"
,"o_HIB"
,"o_HLN"
,"o_HNL"
,"o_HOB"
,"o_HOU"
,"o_HPN"
,"o_HRL"
,"o_HSV"
,"o_HTS"
,"o_HVN"
,"o_HYA"
,"o_HYS"
,"o_IAD"
,"o_IAG"
,"o_IAH"
,"o_ICT"
,"o_IDA"
,"o_ILG"
,"o_ILM"
,"o_IMT"
,"o_IND"
,"o_INL"
,"o_IPT"
,"o_ISN"
,"o_ISP"
,"o_ITH"
,"o_ITO"
,"o_JAC"
,"o_JAN"
,"o_JAX"
,"o_JFK"
,"o_JHM"
,"o_JLN"
,"o_JMS"
,"o_JNU"
,"o_JST"
,"o_KOA"
,"o_KTN"
,"o_LAN"
,"o_LAR"
,"o_LAS"
,"o_LAW"
,"o_LAX"
,"o_LBB"
,"o_LBE"
,"o_LBF"
,"o_LBL"
,"o_LCH"
,"o_LCK"
,"o_LEX"
,"o_LFT"
,"o_LGA"
,"o_LGB"
,"o_LIH"
,"o_LIT"
,"o_LNK"
,"o_LNY"
,"o_LRD"
,"o_LSE"
,"o_LWB"
,"o_LWS"
,"o_LYH"
,"o_MAF"
,"o_MBS"
,"o_MCI"
,"o_MCO"
,"o_MCW"
,"o_MDT"
,"o_MDW"
,"o_MEI"
,"o_MEM"
,"o_MFE"
,"o_MFR"
,"o_MGM"
,"o_MHK"
,"o_MHT"
,"o_MIA"
,"o_MKE"
,"o_MKG"
,"o_MKK"
,"o_MLB"
,"o_MLI"
,"o_MLU"
,"o_MMH"
,"o_MOB"
,"o_MOT"
,"o_MQT"
,"o_MRY"
,"o_MSN"
,"o_MSO"
,"o_MSP"
,"o_MSY"
,"o_MTJ"
,"o_MVY"
,"o_MYR"
,"o_OAJ"
,"o_OAK"
,"o_OGD"
,"o_OGG"
,"o_OGS"
,"o_OKC"
,"o_OMA"
,"o_OME"
,"o_ONT"
,"o_ORD"
,"o_ORF"
,"o_ORH"
,"o_OTH"
,"o_OTZ"
,"o_OWB"
,"o_PAE"
,"o_PAH"
,"o_PBG"
,"o_PBI"
,"o_PDX"
,"o_PGD"
,"o_PGV"
,"o_PHF"
,"o_PHL"
,"o_PHX"
,"o_PIA"
,"o_PIB"
,"o_PIE"
,"o_PIH"
,"o_PIR"
,"o_PIT"
,"o_PLN"
,"o_PNS"
,"o_PPG"
,"o_PQI"
,"o_PRC"
,"o_PSC"
,"o_PSE"
,"o_PSG"
,"o_PSM"
,"o_PSP"
,"o_PUB"
,"o_PUW"
,"o_PVD"
,"o_PVU"
,"o_PWM"
,"o_RAP"
,"o_RDD"
,"o_RDM"
,"o_RDU"
,"o_RFD"
,"o_RHI"
,"o_RIC"
,"o_RIW"
,"o_RKS"
,"o_RNO"
,"o_ROA"
,"o_ROC"
,"o_ROP"
,"o_ROW"
,"o_RST"
,"o_RSW"
,"o_SAF"
,"o_SAN"
,"o_SAT"
,"o_SAV"
,"o_SBA"
,"o_SBN"
,"o_SBP"
,"o_SBY"
,"o_SCC"
,"o_SCE"
,"o_SCK"
,"o_SDF"
,"o_SEA"
,"o_SFB"
,"o_SFO"
,"o_SGF"
,"o_SGU"
,"o_SHD"
,"o_SHR"
,"o_SHV"
,"o_SIT"
,"o_SJC"
,"o_SJT"
,"o_SJU"
,"o_SLC"
,"o_SLN"
,"o_SMF"
,"o_SMX"
,"o_SNA"
,"o_SPI"
,"o_SPN"
,"o_SPS"
,"o_SRQ"
,"o_STC"
,"o_STL"
,"o_STS"
,"o_STT"
,"o_STX"
,"o_SUN"
,"o_SUX"
,"o_SWF"
,"o_SWO"
,"o_SYR"
,"o_TBN"
,"o_TLH"
,"o_TOL"
,"o_TPA"
,"o_TRI"
,"o_TTN"
,"o_TUL"
,"o_TUS"
,"o_TVC"
,"o_TWF"
,"o_TXK"
,"o_TYR"
,"o_TYS"
,"o_UIN"
,"o_USA"
,"o_VCT"
,"o_VEL"
,"o_VLD"
,"o_VPS"
,"o_WRG"
,"o_WYS"
,"o_XNA"
,"o_XWA"
,"o_YAK"
,"o_YKM"
,"o_YNG"
,"o_YUM"
,"d_ABE"
,"d_ABI"
,"d_ABQ"
,"d_ABR"
,"d_ABY"
,"d_ACK"
,"d_ACT"
,"d_ACV"
,"d_ACY"
,"d_ADK"
,"d_ADQ"
,"d_AEX"
,"d_AGS"
,"d_AKN"
,"d_ALB"
,"d_ALO"
,"d_ALS"
,"d_ALW"
,"d_AMA"
,"d_ANC"
,"d_APN"
,"d_ART"
,"d_ASE"
,"d_ATL"
,"d_ATW"
,"d_ATY"
,"d_AUS"
,"d_AVL"
,"d_AVP"
,"d_AZA"
,"d_AZO"
,"d_BDL"
,"d_BET"
,"d_BFF"
,"d_BFL"
,"d_BFM"
,"d_BGM"
,"d_BGR"
,"d_BHM"
,"d_BIH"
,"d_BIL"
,"d_BIS"
,"d_BJI"
,"d_BKG"
,"d_BLI"
,"d_BLV"
,"d_BMI"
,"d_BNA"
,"d_BOI"
,"d_BOS"
,"d_BPT"
,"d_BQK"
,"d_BQN"
,"d_BRD"
,"d_BRO"
,"d_BRW"
,"d_BTM"
,"d_BTR"
,"d_BTV"
,"d_BUF"
,"d_BUR"
,"d_BWI"
,"d_BZN"
,"d_CAE"
,"d_CAK"
,"d_CDB"
,"d_CDC"
,"d_CDV"
,"d_CGI"
,"d_CHA"
,"d_CHO"
,"d_CHS"
,"d_CID"
,"d_CIU"
,"d_CKB"
,"d_CLE"
,"d_CLL"
,"d_CLT"
,"d_CMH"
,"d_CMI"
,"d_CMX"
,"d_CNY"
,"d_COD"
,"d_COS"
,"d_COU"
,"d_CPR"
,"d_CRP"
,"d_CRW"
,"d_CSG"
,"d_CVG"
,"d_CWA"
,"d_CYS"
,"d_DAB"
,"d_DAL"
,"d_DAY"
,"d_DBQ"
,"d_DCA"
,"d_DDC"
,"d_DEC"
,"d_DEN"
,"d_DFW"
,"d_DHN"
,"d_DIK"
,"d_DLG"
,"d_DLH"
,"d_DRO"
,"d_DRT"
,"d_DSM"
,"d_DTW"
,"d_DUT"
,"d_DVL"
,"d_EAR"
,"d_EAT"
,"d_EAU"
,"d_ECP"
,"d_EGE"
,"d_EKO"
,"d_ELM"
,"d_ELP"
,"d_ERI"
,"d_ESC"
,"d_EUG"
,"d_EVV"
,"d_EWN"
,"d_EWR"
,"d_EYW"
,"d_FAI"
,"d_FAR"
,"d_FAT"
,"d_FAY"
,"d_FCA"
,"d_FLG"
,"d_FLL"
,"d_FLO"
,"d_FNT"
,"d_FOD"
,"d_FSD"
,"d_FSM"
,"d_FWA"
,"d_GCC"
,"d_GCK"
,"d_GEG"
,"d_GFK"
,"d_GGG"
,"d_GJT"
,"d_GNV"
,"d_GPT"
,"d_GRB"
,"d_GRI"
,"d_GRK"
,"d_GRR"
,"d_GSO"
,"d_GSP"
,"d_GST"
,"d_GTF"
,"d_GTR"
,"d_GUC"
,"d_GUM"
,"d_HDN"
,"d_HGR"
,"d_HHH"
,"d_HIB"
,"d_HLN"
,"d_HNL"
,"d_HOB"
,"d_HOU"
,"d_HPN"
,"d_HRL"
,"d_HSV"
,"d_HTS"
,"d_HVN"
,"d_HYA"
,"d_HYS"
,"d_IAD"
,"d_IAG"
,"d_IAH"
,"d_ICT"
,"d_IDA"
,"d_ILG"
,"d_ILM"
,"d_IMT"
,"d_IND"
,"d_INL"
,"d_IPT"
,"d_ISN"
,"d_ISP"
,"d_ITH"
,"d_ITO"
,"d_JAC"
,"d_JAN"
,"d_JAX"
,"d_JFK"
,"d_JHM"
,"d_JLN"
,"d_JMS"
,"d_JNU"
,"d_JST"
,"d_KOA"
,"d_KTN"
,"d_LAN"
,"d_LAR"
,"d_LAS"
,"d_LAW"
,"d_LAX"
,"d_LBB"
,"d_LBE"
,"d_LBF"
,"d_LBL"
,"d_LCH"
,"d_LCK"
,"d_LEX"
,"d_LFT"
,"d_LGA"
,"d_LGB"
,"d_LIH"
,"d_LIT"
,"d_LNK"
,"d_LNY"
,"d_LRD"
,"d_LSE"
,"d_LWB"
,"d_LWS"
,"d_LYH"
,"d_MAF"
,"d_MBS"
,"d_MCI"
,"d_MCO"
,"d_MCW"
,"d_MDT"
,"d_MDW"
,"d_MEI"
,"d_MEM"
,"d_MFE"
,"d_MFR"
,"d_MGM"
,"d_MHK"
,"d_MHT"
,"d_MIA"
,"d_MKE"
,"d_MKG"
,"d_MKK"
,"d_MLB"
,"d_MLI"
,"d_MLU"
,"d_MMH"
,"d_MOB"
,"d_MOT"
,"d_MQT"
,"d_MRY"
,"d_MSN"
,"d_MSO"
,"d_MSP"
,"d_MSY"
,"d_MTJ"
,"d_MVY"
,"d_MYR"
,"d_OAJ"
,"d_OAK"
,"d_OGD"
,"d_OGG"
,"d_OGS"
,"d_OKC"
,"d_OMA"
,"d_OME"
,"d_ONT"
,"d_ORD"
,"d_ORF"
,"d_ORH"
,"d_OTH"
,"d_OTZ"
,"d_OWB"
,"d_PAE"
,"d_PAH"
,"d_PBG"
,"d_PBI"
,"d_PDX"
,"d_PGD"
,"d_PGV"
,"d_PHF"
,"d_PHL"
,"d_PHX"
,"d_PIA"
,"d_PIB"
,"d_PIE"
,"d_PIH"
,"d_PIR"
,"d_PIT"
,"d_PLN"
,"d_PNS"
,"d_PPG"
,"d_PQI"
,"d_PRC"
,"d_PSC"
,"d_PSE"
,"d_PSG"
,"d_PSM"
,"d_PSP"
,"d_PUB"
,"d_PUW"
,"d_PVD"
,"d_PVU"
,"d_PWM"
,"d_RAP"
,"d_RDD"
,"d_RDM"
,"d_RDU"
,"d_RFD"
,"d_RHI"
,"d_RIC"
,"d_RIW"
,"d_RKS"
,"d_RNO"
,"d_ROA"
,"d_ROC"
,"d_ROP"
,"d_ROW"
,"d_RST"
,"d_RSW"
,"d_SAF"
,"d_SAN"
,"d_SAT"
,"d_SAV"
,"d_SBA"
,"d_SBN"
,"d_SBP"
,"d_SBY"
,"d_SCC"
,"d_SCE"
,"d_SCK"
,"d_SDF"
,"d_SEA"
,"d_SFB"
,"d_SFO"
,"d_SGF"
,"d_SGU"
,"d_SHD"
,"d_SHR"
,"d_SHV"
,"d_SIT"
,"d_SJC"
,"d_SJT"
,"d_SJU"
,"d_SLC"
,"d_SLN"
,"d_SMF"
,"d_SMX"
,"d_SNA"
,"d_SPI"
,"d_SPN"
,"d_SPS"
,"d_SRQ"
,"d_STC"
,"d_STL"
,"d_STS"
,"d_STT"
,"d_STX"
,"d_SUN"
,"d_SUX"
,"d_SWF"
,"d_SWO"
,"d_SYR"
,"d_TBN"
,"d_TLH"
,"d_TOL"
,"d_TPA"
,"d_TRI"
,"d_TTN"
,"d_TUL"
,"d_TUS"
,"d_TVC"
,"d_TWF"
,"d_TXK"
,"d_TYR"
,"d_TYS"
,"d_UIN"
,"d_USA"
,"d_VCT"
,"d_VEL"
,"d_VLD"
,"d_VPS"
,"d_WRG"
,"d_WYS"
,"d_XNA"
,"d_XWA"
,"d_YAK"
,"d_YKM"
,"d_YNG"
,"d_YUM"
,"a_AA"
,"a_AS"
,"a_B6"
,"a_DL"
,"a_F9"
,"a_G4"
,"a_HA"
,"a_NK"
,"a_UA"
,"a_VX"
,"a_WN"]

################## VALIDATIONS #################

env_valid_options = dict(
    DATASET_SIZE=["1k", "10k", "100k", "500k", "50M", "new", "full", "6m","14m"],
    VALIDATION_DATASET_SIZE=["1k", "10k", "100k", "500k", "500k", "new", "full", "6m"],
    DATA_SOURCE=["local", "big query"],
    MODEL_TARGET=["local", "gcs", "mlflow"],
)

def validate_env_value(env, valid_options):
    env_value = os.environ[env]
    if env_value not in valid_options:
        raise NameError(f"Invalid value for {env} in `.env` file: {env_value} must be in {valid_options}")


for env, valid_options in env_valid_options.items():
    validate_env_value(env, valid_options)
