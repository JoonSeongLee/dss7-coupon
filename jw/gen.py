from ml_config import *

def coupon():
    return pd.read_csv('../data/coupon_list_train.csv',memory_map=True,index_col=0)

def user():
    return pd.read_csv('../data/user_list.csv',memory_map=True,index_col=0)

def view(reset_index=True):
    tmp = pd.read_csv('../data/coupon_visit_train.csv',memory_map=True,index_col=0)
    return tmp.reset_index() if reset_index else tmp

def purchase():
    return pd.read_csv('../data/coupon_detail_train.csv',memory_map=True,index_col=0)

def area():
    return pd.read_csv('../data/coupon_area_train.csv',memory_map=True,index_col=0)

def pref():
    return pd.read_csv('../data/prefecture_locations.csv',memory_map=True,index_col=0)

# --------------------------------------------------------------------------------------------

# version 1 for coupon_list
def coupon_pref():
    return pd.merge(coupon(), pref(), left_on='ken_name', right_on='PREF_NAME')

# version 2 for coupon_list
def coupon_area_pref():
    return pd.merge(coupon_pref(), area(), left_on=['COUPON_ID_hash','PREF_NAME','small_area_name'], right_on=['COUPON_ID_hash','PREF_NAME','SMALL_AREA_NAME'])

def view_purchase():
    return pd.merge(view(), purchase(), on=['I_DATE', 'PURCHASEID_hash', 'USER_ID_hash'])

# --------------------------------------------------------------------------------------------

def user_view_coupon_pref():
    uv = pd.merge(user(),view())
    return pd.merge(uv, coupon_pref(), left_on='VIEW_COUPON_ID_hash', right_on='COUPON_ID_hash')

def user_view_coupon_area_pref():
    uv = pd.merge(user(),view())
    return pd.merge(uv, coupon_area_pref(), left_on='VIEW_COUPON_ID_hash', right_on='COUPON_ID_hash')

def user_purchase_coupon_pref():
    up = pd.merge(user(), purchase())
    return pd.merge(up, coupon_pref(), on='COUPON_ID_hash')

def user_purchase_coupon_area_pref():
    up = pd.merge(user(), purchase())
    return pd.merge(up, coupon_area_pref(), on='COUPON_ID_hash')

def user_vp_coupon_pref():
    uvp = pd.merge(user(),view_purchase(), on='USER_ID_hash')
    return pd.merge(uvp, coupon_pref(), left_on='VIEW_COUPON_ID_hash',right_on='COUPON_ID_hash')

def user_vp_coupon_area_pref():
    uvp = pd.merge(user(),view_purchase(), on='USER_ID_hash')
    return pd.merge(uvp, coupon_area_pref(), left_on='VIEW_COUPON_ID_hash',right_on='COUPON_ID_hash')