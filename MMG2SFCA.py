import geopandas as gpd
import numpy as np
import pandas as pd
import math

def twoSFCA(matrix_file, maxtime, field_name):
    global supply_shp_file, demand_shp_file, result_file
    global supply_field, demand_field, id_field
    # 读取供应点和需求点的shp文件
    supply_shp = gpd.read_file(supply_shp_file)
    demand_shp = gpd.read_file(demand_shp_file)

    # 读取距离矩阵npy文件
    distance_matrix = np.load(matrix_file)
    '''
    for idx1, point1 in supply_points.iterrows():
        for idx2, point2 in demand_points:
    '''

    demand_weights = distance_matrix.copy().astype(int)
    demand_weights[(demand_weights < 0) | (demand_weights > maxtime)] = 0
    demand_weights[demand_weights > 0] = 1

    supply_columns = supply_shp[[supply_field]]
    demand_columns = demand_shp[[demand_field]]
    supply_pid_columns = supply_shp[[id_field]]
    demand_pid_columns = demand_shp[[id_field]]
    supply_np = supply_columns.to_numpy()
    demand_np = demand_columns.to_numpy().T
    demand_matrix = demand_weights * demand_np
    sum_demand = np.sum(demand_matrix, axis=1)
    # 计算供需比R
    #R_col = supply_np / sum_demand.reshape(sum_demand.shape[0], 1)
    R_col = (supply_np.T / sum_demand).T
    # 计算需求点的可达性A
    R_matrix = demand_weights * R_col
    A_col = np.sum(R_matrix, axis=0)

    R_name = f'R_{field_name}'
    A_name = f'A_{field_name}'

    # pid列和结果列拼接，保存为csv文件
    # supply_columns.rename(columns={supply_field: R_name})
    csv_pid_R = pd.concat([supply_pid_columns, pd.DataFrame(R_col, columns=[R_name])], axis=1)
    csv_pid_A = pd.concat([demand_pid_columns, pd.DataFrame(A_col, columns=[R_name])], axis=1)
    csv_R_file = f"{result_file}\\2SFCA_{field_name}_R.csv"
    csv_A_file = f"{result_file}\\2SFCA_{field_name}_A.csv"
    csv_pid_R.to_csv(csv_R_file, index=False, header=True, encoding='utf_8')
    csv_pid_A.to_csv(csv_A_file, index=False, header=True, encoding='utf_8')

    # 更新供应点的shp文件
    supply_shp[R_name] = R_col
    supply_shp.to_file(supply_shp_file, encoding='gb18030')

    # 更新需求点的shp文件
    demand_shp[A_name] = A_col
    demand_shp.to_file(demand_shp_file, encoding='gb18030')

    return True

def G2SFCA(matrix_file, d0, beta, field_name, normalize = False, sec2hour=False, newfx=False, ffx=False, fx='G'):
    #supply_shp_file, demand_shp_file, distance_matrix_file = file_list
    global supply_shp_file, demand_shp_file, result_file
    global supply_field, demand_field, id_field
    # 读取供应点和需求点的shp文件
    supply_shp = gpd.read_file(supply_shp_file)
    demand_shp = gpd.read_file(demand_shp_file)
    #result_polygons = gpd.read_file(result_polygons_file)


    #id_demand = supply_points.columns.get_loc(supply_field)
    #id_supply = demand_points.columns.get_loc(demand_field)
    # 读取距离矩阵npy文件
    distance_matrix = np.load(matrix_file)
    '''
    for idx1, point1 in supply_points.iterrows():
        for idx2, point2 in demand_points:
    '''

    distance_matrix[(distance_matrix < 0) | (distance_matrix > d0)] = 0
    demand_weights = distance_matrix.copy().astype(int)
    demand_weights[demand_weights > 0] = 1

    # 转化为小时单位
    if sec2hour:
        distance_matrix = distance_matrix.astype(np.double) / 3600

    supply_columns = supply_shp[[supply_field]]
    demand_columns = demand_shp[[demand_field]]
    supply_pid_columns = supply_shp[[id_field]]
    demand_pid_columns = demand_shp[[id_field]]
    supply_np = supply_columns.to_numpy()
    demand_np = demand_columns.to_numpy().T
    # 计算f(d_jk)
    if newfx:
        unit_dis = 3600 / d0 if sec2hour else 1 / d0
        temp_matrix = distance_matrix * unit_dis + 1
    else:
        temp_matrix = np.where(distance_matrix == 0, 1, distance_matrix)
    if fx == 'Ga':
        # 高斯型
        d0_Ga = d0 / 3600 if sec2hour else d0
        d0_Inverse = 1 / d0_Ga
        numerator = np.exp(-0.5 * (distance_matrix * d0_Inverse) ** 2) - np.exp(-0.5)
        denominator = 1 - np.exp(-0.5)
        fx_matrix = np.where(distance_matrix != 0, numerator / denominator, 0)
    elif fx == 'K':
        # 核密度型
        d0_k = d0 / 3600 if sec2hour else d0
        d0_Inverse = 1 / d0_k
        fx_matrix = np.where(distance_matrix != 0, 3 / 4 * (1 - (distance_matrix * d0_Inverse) ** 2), 0)
    else:
        # 重力型
        fx_matrix = np.where(distance_matrix != 0, np.power(temp_matrix, -beta), 0)

    demand_matrix = demand_np * fx_matrix
    sum_demand = np.sum(demand_matrix, axis=1)

    # 计算供需比R
    #R_col = supply_np / sum_demand.reshape(sum_demand.shape[0], 1)
    R_col = (supply_np.T / sum_demand).T
    # 计算需求点的可达性A
    if ffx:
        R_matrix = R_col * fx_matrix * fx_matrix
    else:
        R_matrix = R_col * fx_matrix
    A_col_before = np.sum(R_matrix, axis=0)

    if normalize:
        A_min = np.min(A_col_before)
        _range = np.max(A_col_before) - A_min
        A_col = (A_col_before - A_min) / _range
    else:
        A_col = A_col_before

    A_mean = np.mean(A_col)
    SPAR = A_col/A_mean

    R_name = f'R_{field_name}'
    A_name = f'A_{field_name}'
    SPAR_name = f'S{field_name}'

    # pid列和结果列拼接，保存为csv文件
    # supply_columns.rename(columns={supply_field: R_name})
    csv_pid_R = pd.concat([supply_pid_columns, pd.DataFrame(R_col, columns=[R_name])], axis=1)
    csv_pid_A = pd.concat([demand_pid_columns, pd.DataFrame(A_col, columns=[R_name])], axis=1)
    csv_pid_S = pd.concat([demand_pid_columns, pd.DataFrame(SPAR, columns=[R_name])], axis=1)
    csv_R_file = f"{result_file}\\2SFCA_{field_name}_R.csv"
    csv_A_file = f"{result_file}\\2SFCA_{field_name}_A.csv"
    csv_S_file = f"{result_file}\\2SFCA_{field_name}_S.csv"
    csv_pid_R.to_csv(csv_R_file, index=False, header=True, encoding='utf_8')
    csv_pid_A.to_csv(csv_A_file, index=False, header=True, encoding='utf_8')
    csv_pid_S.to_csv(csv_S_file, index=False, header=True, encoding='utf_8')

    # 更新供应点的shp文件
    supply_shp[R_name] = R_col
    supply_shp.to_file(supply_shp_file, encoding='gb18030')

    # 更新需求点的shp文件
    demand_shp[A_name] = A_col
    demand_shp[SPAR_name] = SPAR
    demand_shp.to_file(demand_shp_file, encoding='gb18030')

    return True

def MM2SFCA(d0, beta, field_name, normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=False):
    #supply_shp_file, demand_shp_file, distance_matrix_file = file_list
    global o_distance_file, result_file, npy_file, supply_shp_file, demand_shp_file
    global supply_field, demand_field, id_field
    # 读取供应点和需求点的shp文件
    supply_shp = gpd.read_file(supply_shp_file)
    demand_shp = gpd.read_file(demand_shp_file)
    #result_polygons = gpd.read_file(result_polygons_file)

    ratio_pop = ["ratio_w", "ratio_d", "ratio_b", "ratio_t"]
    travel_modes = ["walking", "driving", "bicycling", "transit"]

    time_matrix_list = []   # 时间矩阵 列表
    ratio_mode_list = [] #人口比例列 列表
    demand_mode_list = []    #需求人口列 列表
    fx_list = []    # f(dx) 距离衰减列表
    demand_fx_list = []    # 各模式

    supply_columns = supply_shp[[supply_field]]
    demand_columns = demand_shp[[demand_field]]
    supply_pid_columns = supply_shp[[id_field]]
    demand_pid_columns = demand_shp[[id_field]]

    # 供应、需求列
    supply_np = supply_columns.to_numpy()
    demand_np = demand_columns.to_numpy().T
    #demand_sum = np.full((len(demand_columns), 1), 0, dtype=np.double)

    for i in range(len(travel_modes)):
        mode = travel_modes[i]
        mode_npy_filepath = f"{npy_file}\\{mode}"
        #file_distance_matrix = f"{mode_npy_filepath}\\distance_matrix.npy"
        file_time_matrix = f"{mode_npy_filepath}\\time_matrix.npy"
        #file_cost_matrix = f"{mode_npy_filepath}\\cost_matrix.npy"

        #distance_matrix = np.load(file_distance_matrix)
        #cost_matrix = np.load(file_cost_matrix)

        # 时间列提取
        time_matrix = np.load(file_time_matrix)

        # 转化为小时单位
        if sec2hour:
            time_matrix = time_matrix.astype(np.double) / 3600

        time_matrix_list.append(time_matrix)

        # 时间二分矩阵   d_{kj}(T_n) <= d_0(T_n)
        time_matrix[(time_matrix < 0) | (time_matrix > d0)] = 0
        demand_weights = time_matrix.copy()
        demand_weights[demand_weights > 0] = 1

        # 当前模式需求人口计算   t_{n,k} * D_k
        ratio_mode = demand_shp[[ratio_pop[i]]].to_numpy().T
        demand_mode = demand_np * ratio_mode

        # 计算f(d_jk)
        if newfx:
            unit_dis = 3600 / d0 if sec2hour else 1 / d0
            temp_matrix = time_matrix * unit_dis + 1
        else:
            temp_matrix = np.where(time_matrix == 0, 1, time_matrix)

        if fx1:
            fx_matrix = demand_weights
        else:
            # 重力型
            fx_matrix = np.where(time_matrix != 0, np.power(temp_matrix, -beta), 0)

        # 计算 t_{n,k} * D_k * f(d_{kj})
        demand_matrix = demand_mode * fx_matrix

        #sum_demand = np.sum(demand_matrix, axis=1)
        #demand_sum = demand_sum + (supply_np.T / sum_demand).T

        # 添加到list合集
        ratio_mode_list.append(ratio_mode)
        demand_mode_list.append(demand_mode)
        fx_list.append(fx_matrix)
        demand_fx_list.append(demand_matrix)

    # 计算供需比R
    # k∈{dkj(Tn)≤d0(Tn)},t_{n,k} * D_k
    sum_demand_matrix = np.sum(np.array(demand_fx_list), axis=0)
    #
    sum_demand = np.sum(sum_demand_matrix, axis=1)
    R_col = (supply_np.T / sum_demand).T

    # 计算需求点的可达性A
    # t_{n,k} * f(d)
    fx_w_matrix = np.sum(np.array(fx_list) * np.array(ratio_mode_list), axis=0)
    # t_{n,k} * R_j * f(d)
    if ffx:
        R_matrix = R_col * fx_w_matrix * fx_w_matrix
    else:
        R_matrix = R_col * fx_w_matrix
    #A_col_before = np.sum(R_matrix, axis=0)
    A_col_before = np.sum(R_matrix, axis=0)

    if normalize:
        A_min = np.min(A_col_before)
        _range = np.max(A_col_before) - A_min
        A_col = (A_col_before - A_min) / _range
    else:
        A_col = A_col_before

    A_mean = np.mean(A_col)
    SPAR = A_col/A_mean

    R_name = f'R{field_name}'
    A_name = f'A{field_name}'
    SPAR_name = f'S{field_name}'

    # pid列和结果列拼接，保存为csv文件
    # supply_columns.rename(columns={supply_field: R_name})
    csv_pid_R = pd.concat([supply_pid_columns, pd.DataFrame(R_col, columns=[R_name])], axis=1)
    csv_pid_A = pd.concat([demand_pid_columns, pd.DataFrame(A_col, columns=[R_name])], axis=1)
    csv_pid_S = pd.concat([demand_pid_columns, pd.DataFrame(SPAR, columns=[R_name])], axis=1)
    csv_R_file = f"{result_file}\\2SFCA_{field_name}_R.csv"
    csv_A_file = f"{result_file}\\2SFCA_{field_name}_A.csv"
    csv_S_file = f"{result_file}\\2SFCA_{field_name}_S.csv"
    csv_pid_R.to_csv(csv_R_file, index=False, header=True, encoding='utf_8')
    csv_pid_A.to_csv(csv_A_file, index=False, header=True, encoding='utf_8')
    csv_pid_S.to_csv(csv_S_file, index=False, header=True, encoding='utf_8')

    # 更新供应点的shp文件
    supply_shp[R_name] = R_col
    supply_shp.to_file(supply_shp_file, encoding='gb18030')

    # 更新需求点的shp文件
    demand_shp[A_name] = A_col
    demand_shp[SPAR_name] = SPAR
    demand_shp.to_file(demand_shp_file, encoding='gb18030')

    return True


# 读取两个shp文件中的点数据
filepath = r"E:\data\Accessbility"
shp_file1 = '深圳市公立医院'
shp_file2 = '研究单元_点'
shp_file3 = '研究单元_面'
npy_file = '\\'.join([filepath, 'AmapAPI', f"[{shp_file1}][{shp_file2}]"])

id_field = 'pid'

# 文件地址、字段设置
supply_shp_file = f"{filepath}\\{shp_file1}.shp"
demand_shp_file = f"{filepath}\\{shp_file3}.shp"
o_distance_file = f"{npy_file}\\o_distance_matrix.npy"
driving_time_file = f"{npy_file}\\driving\\time_matrix.npy"
#result_polygons_file = f"{filepath}\\{shp_file3}.shp"
supply_field, demand_field = "outpatient", "Pop"
result_file = f"{npy_file}\\A_R_result"

bool_done = False

if bool_done:
    G2SFCA(driving_time_file, 5400, 2.0, 'F_15_2', normalize=False, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'F0_15_2', normalize=True, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'F1_15_2', normalize=False, sec2hour=True, newfx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'F2_15_2', normalize=True, sec2hour=True, newfx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'FF_15_2', normalize=False, sec2hour=False, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'FF0_15_2', normalize=True, sec2hour=False, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'FF1_15_2', normalize=False, sec2hour=True, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'FF2_15_2', normalize=True, sec2hour=True, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'G_15_2', normalize=False, sec2hour=False, newfx=True, ffx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'G0_15_2', normalize=True, sec2hour=False, newfx=True, ffx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'G1_15_2', normalize=False, sec2hour=True, newfx=True, ffx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'G2_15_2', normalize=True, sec2hour=True, newfx=True, ffx=False)
    G2SFCA(driving_time_file, 5400, 2.0, 'GG_15_2', normalize=False, sec2hour=False, newfx=True, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'GG0_15_2', normalize=True, sec2hour=False, newfx=True, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'GG1_15_2', normalize=False, sec2hour=True, newfx=True, ffx=True)
    G2SFCA(driving_time_file, 5400, 2.0, 'GG2_15_2', normalize=True, sec2hour=True, newfx=True, ffx=True)

if bool_done:
    G2SFCA(driving_time_file, 3600, 1.0, 'F_1_1', normalize=False, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'F0_1_1', normalize=True, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'F1_1_1', normalize=False, sec2hour=True, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'F2_1_1', normalize=True, sec2hour=True, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'FF_1_1', normalize=False, sec2hour=False, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 3600, 1.0, 'FF0_1_1', normalize=True, sec2hour=False, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 3600, 1.0, 'FF1_1_1', normalize=False, sec2hour=True, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 3600, 1.0, 'FF2_1_1', normalize=True, sec2hour=True, newfx=False, ffx=True)

if bool_done:
    G2SFCA(driving_time_file, 3600, 2.0, 'F_1_2', normalize=False, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.5, 'F_1_15', normalize=False, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'F_1_1', normalize=False, sec2hour=False, newfx=False)

if bool_done:
    G2SFCA(driving_time_file, 3600, 2.0, 'FF_1_2', normalize=False, sec2hour=False, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 3600, 1.5, 'FF_1_15', normalize=False, sec2hour=False, newfx=False, ffx=True)
    G2SFCA(driving_time_file, 3600, 1.0, 'FF_1_1', normalize=False, sec2hour=False, newfx=False, ffx=True)

if bool_done:
    MM2SFCA(3600, 2.0, 'M_1_2', normalize=False, sec2hour=False, newfx=False, fx1=True, ffx=False)
    MM2SFCA(3600, 1.5, 'M_1_15', normalize=False, sec2hour=False, newfx=False, fx1=True, ffx=False)
    MM2SFCA(3600, 1.0, 'M_1_1', normalize=False, sec2hour=False, newfx=False, fx1=True, ffx=False)
    MM2SFCA(3600, 2.0, 'MF_1_2', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=False)
    MM2SFCA(3600, 1.5, 'MF_1_15', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=False)
    MM2SFCA(3600, 1.0, 'MF_1_1', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=False)
    MM2SFCA(3600, 2.0, 'MM_1_2', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=True)
    MM2SFCA(3600, 1.5, 'MM_1_15', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=True)
    MM2SFCA(3600, 1.0, 'MM_1_1', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=True)

if bool_done:
    MM2SFCA(3600, 2.0, 'MG_1_2', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=False)
    MM2SFCA(3600, 1.5, 'MG_1_15', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=False)
    MM2SFCA(3600, 1.0, 'MG_1_1', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=False)
    MM2SFCA(3600, 2.0, 'MGG_1_2', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 1.5, 'MGG_1_15', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 1.0, 'MGG_1_1', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)

if bool_done:
    G2SFCA(driving_time_file, 3600, 2.0, 'K_1', normalize=False, sec2hour=False, newfx=False, fx='K')
    G2SFCA(driving_time_file, 3600, 2.0, 'Ga_1', normalize=False, sec2hour=False, newfx=False, fx='Ga')

if bool_done:
    MM2SFCA(3600, 1.0, 'M_1_1', normalize=False, sec2hour=False, newfx=False, fx1=True, ffx=False)
    MM2SFCA(3600, 1.0, 'M0_1_1', normalize=True, sec2hour=False, newfx=False, fx1=True, ffx=False)
    MM2SFCA(3600, 1.0, 'MM_1_1', normalize=False, sec2hour=False, newfx=False, fx1=False, ffx=True)
    MM2SFCA(3600, 1.0, 'MM0_1_1', normalize=True, sec2hour=False, newfx=False, fx1=False, ffx=True)
    MM2SFCA(3600, 1.0, 'MGG_1_1', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 1.0, 'MGG0_1_1', normalize=True, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 0.9, 'MGG_1_09', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 0.9, 'MGG0_1_09', normalize=True, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 0.5, 'MGG_1_05', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 0.5, 'MGG0_1_05', normalize=True, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 1.5, 'MGG_1_15', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 1.5, 'MGG0_1_15', normalize=True, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 2.0, 'MGG_1_2', normalize=False, sec2hour=False, newfx=True, fx1=False, ffx=True)
    MM2SFCA(3600, 2.0, 'MGG0_1_2', normalize=True, sec2hour=False, newfx=True, fx1=False, ffx=True)

if bool_done:
    G2SFCA(driving_time_file, 3600, 1.0, 'F_1_1', normalize=False, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'F0_1_1', normalize=True, sec2hour=False, newfx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'G_1_1', normalize=False, sec2hour=False, newfx=True, ffx=False)
    G2SFCA(driving_time_file, 3600, 1.0, 'G0_1_1', normalize=True, sec2hour=False, newfx=True, ffx=False)
    G2SFCA(driving_time_file, 3600, 1.0, '')

if True:
    G2SFCA(driving_time_file, 3600, 1.0, 'T1_1_1', normalize=False, sec2hour=False, newfx=False)
    MM2SFCA(3600, 1.0, 'T2_1_1', normalize=False, sec2hour=False, newfx=False, fx1=True, ffx=False)
