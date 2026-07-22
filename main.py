import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(page_title="Digital Beamforming Simulator", layout="wide")

# 세션 상태 변수 초기화 (자동 빔 스캔용)
if "is_scanning" not in st.session_state:
    st.session_state.is_scanning = False
if "scan_idx" not in st.session_state:
    st.session_state.scan_idx = 0

st.title("📡 디지털 빔포밍 시뮬레이터 v1.0")
st.markdown("주파수, 안테나 배열, 조향 각도(Azimuth/Elevation)를 조절하여 빔 패턴과 안테나 상태를 확인하세요.")

# --- 사이드바: 입력 파라미터 ---
st.sidebar.header("⚙️ 시뮬레이터 입력 설정")

freq_ghz = st.sidebar.slider("주파수 (GHz)", min_value=1.0, max_value=60.0, value=28.0, step=0.5)
M = st.sidebar.slider("수직 안테나 수 (M)", min_value=1, max_value=128, value=4, step=1)
N = st.sidebar.slider("수평 안테나 수 (N)", min_value=1, max_value=128, value=4, step=1)
d_lambda = st.sidebar.slider("안테나 간격 (파장 기준, d/λ)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

azimuth_deg = st.sidebar.slider("목표 Azimuth 각도 (°)", min_value=-90.0, max_value=90.0, value=0.0, step=1.0)
elevation_deg = st.sidebar.slider("목표 Elevation 각도 (°)", min_value=-90.0, max_value=90.0, value=0.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("📊 시각화 설정")
scale_option = st.sidebar.radio(
    "3D 빔 패턴 스케일", 
    options=["dB Scale (추천)", "Linear Scale"], 
    index=0
)
show_3db = st.sidebar.checkbox("2D 패턴 3dB 대역폭 범위 표시", value=True)
show_3db_value = st.sidebar.checkbox("2D 패턴 3dB 대역폭 값 표시", value=True)

st.sidebar.markdown("---")
with st.sidebar.expander("📡 자동 빔 스캔 (Auto Sweep)", expanded=False):
    az_range = st.slider("Azimuth 스캔 범위 (°)", -90.0, 90.0, (-45.0, 45.0), step=1.0)
    el_range = st.slider("Elevation 스캔 범위 (°)", -90.0, 90.0, (-15.0, 15.0), step=1.0)
    az_steps = st.slider("Azimuth 스텝 수", 3, 50, 10, step=1)
    el_steps = st.slider("Elevation 스텝 수", 2, 20, 5, step=1)
    scan_delay = st.slider("프레임 지연 (초)", 0.02, 1.0, 0.1, step=0.02)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶️ 스캔 시작", disabled=st.session_state.is_scanning, use_container_width=True):
            st.session_state.is_scanning = True
            st.session_state.scan_idx = 0
            st.rerun()
    with col_btn2:
        if st.button("⏹️ 스캔 중지", disabled=not st.session_state.is_scanning, use_container_width=True):
            st.session_state.is_scanning = False
            st.rerun()

# --- 물리 상수 및 계산 ---
c = 3e8  # 빛의 속도 (m/s)
freq = freq_ghz * 1e9
lam = c / freq
d = d_lambda * lam  # 안테나 물리적 간격 (m)

# 조향 각도 보간 (자동 스캔 활성화 시, 래스터 스캔 방식)
if st.session_state.is_scanning:
    total_steps = az_steps * el_steps
    # 몫과 나머지 연산으로 Elevation과 Azimuth 인덱스 역산 (Elevation 고정 후 Azimuth 스윕)
    el_idx = st.session_state.scan_idx // az_steps
    az_idx = st.session_state.scan_idx % az_steps
    
    t_az = az_idx / (az_steps - 1) if az_steps > 1 else 0.0
    t_el = el_idx / (el_steps - 1) if el_steps > 1 else 0.0
    
    current_azimuth = az_range[0] + t_az * (az_range[1] - az_range[0])
    current_elevation = el_range[0] + t_el * (el_range[1] - el_range[0])
    
    st.sidebar.info(f"🔄 스캔 각도: Az={current_azimuth:.1f}°, El={current_elevation:.1f}° (Step {st.session_state.scan_idx + 1}/{total_steps})")
else:
    current_azimuth = azimuth_deg
    current_elevation = elevation_deg

az = np.radians(current_azimuth)
el = np.radians(current_elevation)

# 조향 벡터 (Steering Vector) 계산
n_indices = np.arange(N) - (N - 1) / 2.0
m_indices = np.arange(M) - (M - 1) / 2.0

# 수평(Y축), 수직(Z축) 조향 위상 변화율 계산 (3D 빔 패턴과 물리 좌표 매치)
k_y = 2 * np.pi / lam * np.cos(el) * np.sin(az)  # 수평 방향 조향 성분
k_z = 2 * np.pi / lam * np.sin(el)               # 수직 방향 조향 성분

# 각 안테나 소자의 위치 및 가중치(위상) 계산
Y, Z = np.meshgrid(n_indices * d, m_indices * d)
phases = - (k_y * Y + k_z * Z)
phases_deg = np.degrees(phases) % 360  # 0~360 도로 표현

# --- 탭 구성 ---
tab1, tab2, tab3 = st.tabs(["📊 빔 패턴 (2D/3D)", "🔍 3dB 대역폭", "🔴 안테나 배치 및 위상"])

with tab1:
    st.subheader("방사 패턴 (Radiation Pattern)")
    
    col1, col2 = st.columns(2)
    
    # 2D Azimuth 컷 패턴 계산 (NumPy 벡터화 적용)
    phi_scan = np.linspace(-np.pi/2, np.pi/2, 360)
    # phi_scan: (360, 1), n_indices: (1, N) 형태로 브로드캐스팅
    steering_matrix = np.exp(1j * 2 * np.pi / lam * d * n_indices[None, :] * (np.sin(phi_scan[:, None]) - np.sin(az)))
    af_2d = np.sum(steering_matrix, axis=1)
    af_2d_db = 20 * np.log10(np.abs(af_2d) / np.max(np.abs(af_2d)) + 1e-6)
    
    # 2D Elevation 컷 패턴 계산 (NumPy 벡터화 적용)
    theta_scan = np.linspace(-np.pi/2, np.pi/2, 360)
    # theta_scan: (360, 1), m_indices: (1, M) 형태로 브로드캐스팅
    steering_matrix_el = np.exp(1j * 2 * np.pi / lam * d * m_indices[None, :] * (np.sin(theta_scan[:, None]) - np.sin(el)))
    af_2d_el = np.sum(steering_matrix_el, axis=1)
    af_2d_el_db = 20 * np.log10(np.abs(af_2d_el) / np.max(np.abs(af_2d_el)) + 1e-6)
    
    # --- First Null 검출을 위한 헬퍼 함수 ---
    def find_first_null(af_db, max_idx):
        null_left = 0
        for i in range(max_idx - 1, 0, -1):
            if af_db[i] > af_db[i+1]:
                null_left = i + 1
                break
        null_right = len(af_db) - 1
        for i in range(max_idx + 1, len(af_db) - 1):
            if af_db[i] > af_db[i-1]:
                null_right = i - 1
                break
        return null_left, null_right

    # --- 3dB 대역폭 공통 인덱스 계산 (Azimuth 컷) ---
    max_idx_az = np.argmax(np.abs(af_2d))
    half_power_az = np.abs(af_2d[max_idx_az]) / np.sqrt(2)
    left_side_az = np.abs(af_2d[:max_idx_az])
    right_side_az = np.abs(af_2d[max_idx_az:])
    
    idx_left_az = None
    idx_right_az = None
    bw_deg_az = 0.0
    
    try:
        left_matches_az = np.where(left_side_az <= half_power_az)[0]
        right_matches_az = np.where(right_side_az <= half_power_az)[0]
        
        if len(left_matches_az) > 0 and len(right_matches_az) > 0:
            idx_left_az = left_matches_az[-1]
            idx_right_az = max_idx_az + right_matches_az[0]
            bw_rad_az = phi_scan[idx_right_az] - phi_scan[idx_left_az]
            bw_deg_az = np.degrees(bw_rad_az)
        elif len(left_matches_az) > 0:
            idx_left_az = left_matches_az[-1]
            bw_rad_az = 2 * (phi_scan[max_idx_az] - phi_scan[idx_left_az])
            bw_deg_az = np.degrees(bw_rad_az)
            idx_right_az = min(len(phi_scan) - 1, max_idx_az + (max_idx_az - idx_left_az))
        elif len(right_matches_az) > 0:
            idx_right_az = max_idx_az + right_matches_az[0]
            bw_rad_az = 2 * (phi_scan[idx_right_az] - phi_scan[max_idx_az])
            bw_deg_az = np.degrees(bw_rad_az)
            idx_left_az = max(0, max_idx_az - (idx_right_az - max_idx_az))
    except:
        pass

    # --- FNBW 계산 (Azimuth 컷) ---
    null_l_az, null_r_az = find_first_null(af_2d_db, max_idx_az)
    if null_l_az == 0 and null_r_az < len(phi_scan) - 1:
        fnbw_deg_az = np.degrees(2 * (phi_scan[max_idx_az] - phi_scan[null_l_az]))
    elif null_r_az == len(phi_scan) - 1 and null_l_az > 0:
        fnbw_deg_az = np.degrees(2 * (phi_scan[null_r_az] - phi_scan[max_idx_az]))
    else:
        fnbw_deg_az = np.degrees(phi_scan[null_r_az] - phi_scan[null_l_az])

    # --- SLL 계산 (Azimuth 컷) ---
    sll_parts_az = []
    sll_indices_az = []
    if null_l_az > 0:
        sll_parts_az.append(af_2d_db[:null_l_az])
        sll_indices_az.append(np.arange(0, null_l_az))
    if null_r_az < len(phi_scan) - 1:
        sll_parts_az.append(af_2d_db[null_r_az+1:])
        sll_indices_az.append(np.arange(null_r_az+1, len(phi_scan)))
    
    if len(sll_parts_az) > 0:
        sll_concat_az = np.concatenate(sll_parts_az)
        sll_idx_concat_az = np.concatenate(sll_indices_az)
        max_sll_idx_az = np.argmax(sll_concat_az)
        
        sll_db_az = sll_concat_az[max_sll_idx_az]
        sll_angle_az = np.degrees(phi_scan[sll_idx_concat_az[max_sll_idx_az]])
    else:
        sll_db_az = -99.0
        sll_angle_az = 0.0

    # --- 3dB 대역폭 공통 인덱스 계산 (Elevation 컷) ---
    max_idx_el = np.argmax(np.abs(af_2d_el))
    half_power_el = np.abs(af_2d_el[max_idx_el]) / np.sqrt(2)
    left_side_el = np.abs(af_2d_el[:max_idx_el])
    right_side_el = np.abs(af_2d_el[max_idx_el:])
    
    idx_left_el = None
    idx_right_el = None
    bw_deg_el = 0.0
    
    try:
        left_matches_el = np.where(left_side_el <= half_power_el)[0]
        right_matches_el = np.where(right_side_el <= half_power_el)[0]
        
        if len(left_matches_el) > 0 and len(right_matches_el) > 0:
            idx_left_el = left_matches_el[-1]
            idx_right_el = max_idx_el + right_matches_el[0]
            bw_rad_el = theta_scan[idx_right_el] - theta_scan[idx_left_el]
            bw_deg_el = np.degrees(bw_rad_el)
        elif len(left_matches_el) > 0:
            idx_left_el = left_matches_el[-1]
            bw_rad_el = 2 * (theta_scan[max_idx_el] - theta_scan[idx_left_el])
            bw_deg_el = np.degrees(bw_rad_el)
            idx_right_el = min(len(theta_scan) - 1, max_idx_el + (max_idx_el - idx_left_el))
        elif len(right_matches_el) > 0:
            idx_right_el = max_idx_el + right_matches_el[0]
            bw_rad_el = 2 * (theta_scan[idx_right_el] - theta_scan[max_idx_el])
            bw_deg_el = np.degrees(bw_rad_el)
            idx_left_el = max(0, max_idx_el - (idx_right_el - max_idx_el))
    except:
        pass
        
    # --- FNBW 계산 (Elevation 컷) ---
    null_l_el, null_r_el = find_first_null(af_2d_el_db, max_idx_el)
    if null_l_el == 0 and null_r_el < len(theta_scan) - 1:
        fnbw_deg_el = np.degrees(2 * (theta_scan[max_idx_el] - theta_scan[null_l_el]))
    elif null_r_el == len(theta_scan) - 1 and null_l_el > 0:
        fnbw_deg_el = np.degrees(2 * (theta_scan[null_r_el] - theta_scan[max_idx_el]))
    else:
        fnbw_deg_el = np.degrees(theta_scan[null_r_el] - theta_scan[null_l_el])
        
    # --- SLL 계산 (Elevation 컷) ---
    sll_parts_el = []
    sll_indices_el = []
    if null_l_el > 0:
        sll_parts_el.append(af_2d_el_db[:null_l_el])
        sll_indices_el.append(np.arange(0, null_l_el))
    if null_r_el < len(theta_scan) - 1:
        sll_parts_el.append(af_2d_el_db[null_r_el+1:])
        sll_indices_el.append(np.arange(null_r_el+1, len(theta_scan)))
        
    if len(sll_parts_el) > 0:
        sll_concat_el = np.concatenate(sll_parts_el)
        sll_idx_concat_el = np.concatenate(sll_indices_el)
        max_sll_idx_el = np.argmax(sll_concat_el)
        
        sll_db_el = sll_concat_el[max_sll_idx_el]
        sll_angle_el = np.degrees(theta_scan[sll_idx_concat_el[max_sll_idx_el]])
    else:
        sll_db_el = -99.0
        sll_angle_el = 0.0
    
    with col1:
        st.markdown("#### 2D Azimuth 빔 패턴 (Polar)")
        fig_az, ax_az = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
        ax_az.plot(phi_scan, af_2d_db, color='dodgerblue')
        ax_az.set_theta_zero_location("N")
        ax_az.set_theta_direction(-1)
        ax_az.set_rlim([-40, 0])
        
        # 3dB 대역폭 범위 표시
        if show_3db:
            ax_az.axhline(-3, color='gray', linestyle=':', alpha=0.7)
            if idx_left_az is not None and idx_right_az is not None:
                phi_l = phi_scan[idx_left_az]
                phi_r = phi_scan[idx_right_az]
                theta_fill = phi_scan[idx_left_az:idx_right_az+1]
                r_fill = af_2d_db[idx_left_az:idx_right_az+1]
                
                # 수치 표시 여부에 따른 범례 텍스트 분기
                lbl = f'3dB BW: {bw_deg_az:.2f}°' if show_3db_value else '3dB Beamwidth'
                # 영역 음영 채우기
                ax_az.fill_between(theta_fill, -40, r_fill, color='dodgerblue', alpha=0.15, label=lbl)
                # 양 끝단 경계선 그리기
                ax_az.plot([phi_l, phi_l], [-40, 0], color='dodgerblue', linestyle='--', linewidth=1)
                ax_az.plot([phi_r, phi_r], [-40, 0], color='dodgerblue', linestyle='--', linewidth=1)
                
        # 범위는 꺼져 있지만 값 표시만 켜져 있는 경우 (가상 범례 사용)
        if not show_3db and show_3db_value:
            if idx_left_az is not None and idx_right_az is not None:
                from matplotlib.patches import Patch
                legend_patch = Patch(color='dodgerblue', alpha=0.0, label=f'3dB BW: {bw_deg_az:.2f}°')
                ax_az.legend(handles=[legend_patch], loc='lower right', bbox_to_anchor=(1.15, -0.15))
        elif show_3db:
            # 범위가 켜져 있고 수치 표시 여부(텍스트 포함 여부)와 무관하게 범례 박스 출력
            ax_az.legend(loc='lower right', bbox_to_anchor=(1.15, -0.15))
                
        ax_az.set_title(f"Azimuth = {current_azimuth:.1f}° (El=0° Cut)", va='bottom')
        st.pyplot(fig_az)

    with col2:
        st.markdown("#### 2D Elevation 빔 패턴 (Polar)")
        fig_el, ax_el = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 5))
        ax_el.plot(theta_scan, af_2d_el_db, color='crimson')
        ax_el.set_theta_zero_location("N")
        ax_el.set_theta_direction(-1)
        ax_el.set_rlim([-40, 0])
        
        # 3dB 대역폭 범위 표시
        if show_3db:
            ax_el.axhline(-3, color='gray', linestyle=':', alpha=0.7)
            if idx_left_el is not None and idx_right_el is not None:
                theta_l = theta_scan[idx_left_el]
                theta_r = theta_scan[idx_right_el]
                theta_fill = theta_scan[idx_left_el:idx_right_el+1]
                r_fill = af_2d_el_db[idx_left_el:idx_right_el+1]
                
                # 수치 표시 여부에 따른 범례 텍스트 분기
                lbl = f'3dB BW: {bw_deg_el:.2f}°' if show_3db_value else '3dB Beamwidth'
                # 영역 음영 채우기 (Elevation은 붉은색)
                ax_el.fill_between(theta_fill, -40, r_fill, color='crimson', alpha=0.15, label=lbl)
                # 양 끝단 경계선 그리기
                ax_el.plot([theta_l, theta_l], [-40, 0], color='crimson', linestyle='--', linewidth=1)
                ax_el.plot([theta_r, theta_r], [-40, 0], color='crimson', linestyle='--', linewidth=1)
                
        # 범위는 꺼져 있지만 값 표시만 켜져 있는 경우 (가상 범례 사용)
        if not show_3db and show_3db_value:
            if idx_left_el is not None and idx_right_el is not None:
                from matplotlib.patches import Patch
                legend_patch = Patch(color='crimson', alpha=0.0, label=f'3dB BW: {bw_deg_el:.2f}°')
                ax_el.legend(handles=[legend_patch], loc='lower right', bbox_to_anchor=(1.15, -0.15))
        elif show_3db:
            # 범위가 켜져 있고 수치 표시 여부(텍스트 포함 여부)와 무관하게 범례 박스 출력
            ax_el.legend(loc='lower right', bbox_to_anchor=(1.15, -0.15))
                
        ax_el.set_title(f"Elevation = {current_elevation:.1f}° (Az=0° Cut)", va='bottom')
        st.pyplot(fig_el)

    st.markdown("---")
    st.markdown("#### 3D 빔 패턴 (Spherical Surface)")
    
    # 3D 구면 좌표계 패턴 벡터화 계산
    theta_3d = np.linspace(0, np.pi, 50)  # 극각
    phi_3d = np.linspace(-np.pi, np.pi, 50)  # 방위각
    THETA, PHI = np.meshgrid(theta_3d, phi_3d)
    
    # 안테나 배열 소자들의 좌표를 그리드로 생성
    Y_pos, Z_pos = np.meshgrid(n_indices * d, m_indices * d)
    Y_flat = Y_pos.ravel()[None, None, :]  # shape: (1, 1, M*N)
    Z_flat = Z_pos.ravel()[None, None, :]  # shape: (1, 1, M*N)
    
    T_grid = THETA[:, :, None]             # shape: (50, 50, 1)
    P_grid = PHI[:, :, None]               # shape: (50, 50, 1)
    
    # 브로드캐스팅 연산을 통해 모든 안테나 소자의 phase shift 한 번에 계산
    phase_shift = 2 * np.pi / lam * (Y_flat * np.sin(T_grid) * np.sin(P_grid) + Z_flat * np.cos(T_grid))
    target_shift = 2 * np.pi / lam * (Y_flat * np.cos(el) * np.sin(az) + Z_flat * np.sin(el))
    
    AF_3D = np.sum(np.exp(1j * (phase_shift - target_shift)), axis=2)
    # 사용자가 선택한 스케일 모드에 따른 3D 빔 렌더링 파라미터 분기
    if "dB Scale" in scale_option:
        # R(방사 강도)을 dB 스케일로 정규화하여 구면 반경으로 활용 (입체 부엽 시각화)
        R_db = 20 * np.log10(np.abs(AF_3D) / np.max(np.abs(AF_3D)) + 1e-6)
        min_db = -30.0  # 표시할 하한선 dB
        R_render = np.clip((R_db - min_db) / (-min_db), 0.0, 1.0)
        hover_text = np.vectorize(lambda db: f"빔 강도: {db:.1f} dB")(R_db)
        title_text = "3D Normalized Beam Pattern (dB Scale, Min -30dB)"
        color_data = R_db
        c_min = -30.0
        c_max = 0.0
        colorbar_title = "강도 (dB)"
    else:
        # Linear Scale 적용
        R_linear = np.abs(AF_3D) / np.max(np.abs(AF_3D))
        R_render = R_linear
        hover_text = np.vectorize(lambda val: f"빔 강도: {val:.3f} (Linear)")(R_linear)
        title_text = "3D Normalized Beam Pattern (Linear Scale)"
        color_data = R_linear
        c_min = 0.0
        c_max = 1.0
        colorbar_title = "강도 (Linear)"
        
    # 표준 구면 좌표 매핑을 적용하여 조향 및 물리 연산과 일치시킴
    X = R_render * np.sin(THETA) * np.cos(PHI)
    Y_coord = R_render * np.sin(THETA) * np.sin(PHI)
    Z_coord = R_render * np.cos(THETA)
    
    fig_3d = go.Figure(data=[go.Surface(
        x=X, y=Y_coord, z=Z_coord, 
        surfacecolor=color_data,  # 표면 색상은 정규화되지 않은 실제 데이터 대입 (범례 스케일 연동)
        colorscale='Viridis', 
        cmin=c_min, 
        cmax=c_max,
        colorbar=dict(title=colorbar_title, thickness=15),
        hovertext=hover_text
    )])
    
    fig_3d.update_layout(
        title=title_text, 
        scene=dict(
            xaxis=dict(title='X (Forward)', range=[-1, 1]),
            yaxis=dict(title='Y (Horizontal)', range=[-1, 1]),
            zaxis=dict(title='Z (Vertical)', range=[-1, 1]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)  # 1:1:1 구형 종횡비 강제
        ),
        margin=dict(l=0, r=0, b=0, t=30), 
        height=550
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    st.subheader("📏 주요 성능 지표 (AESA Performance Metrics)")
    
    # 안테나 배열 게인 계산 (dBi)
    array_gain_db = 10 * np.log10(M * N)
    
    # 이득 메트릭 (1행)
    st.metric(label="배열 안테나 이득 (Array Gain)", value=f"{array_gain_db:.2f} dBi")
    st.markdown("---")
    
    # 3dB 대역폭 (2행)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="계산된 3dB Beamwidth (Azimuth 컷)", value=f"{bw_deg_az:.2f} °")
    with col_m2:
        st.metric(label="계산된 3dB Beamwidth (Elevation 컷)", value=f"{bw_deg_el:.2f} °")
        
    st.markdown("---")
    
    # FNBW (3행)
    col_m3, col_m4 = st.columns(2)
    with col_m3:
        st.metric(label="계산된 First Null Bandwidth (Azimuth)", value=f"{fnbw_deg_az:.2f} °")
    with col_m4:
        st.metric(label="계산된 First Null Bandwidth (Elevation)", value=f"{fnbw_deg_el:.2f} °")
        
    st.markdown("---")
    
    # SLL (4행)
    col_m5, col_m6 = st.columns(2)
    with col_m5:
        val_az = f"{sll_db_az:.2f} dB (@ {sll_angle_az:.1f}°)" if sll_db_az > -90.0 else "N/A (부엽 없음)"
        st.metric(label="부엽 레벨 Sidelobe Level (Azimuth)", value=val_az)
    with col_m6:
        val_el = f"{sll_db_el:.2f} dB (@ {sll_angle_el:.1f}°)" if sll_db_el > -90.0 else "N/A (부엽 없음)"
        st.metric(label="부엽 레벨 Sidelobe Level (Elevation)", value=val_el)
        
    st.info("💡 **안테나 소자 수가 많을수록** 지향성 이득(Gain)이 증가하고 빔 폭(3dB & FNBW)이 좁아져 전파 집중도가 상승합니다. 부엽 레벨(SLL)이 낮을수록 목표 방향 외의 간섭 신호 방출이 최소화됩니다.")

with tab3:
    st.subheader("🔴 안테나 배열 및 각 소자별 위상 (Phase)")
    
    if M * N > 1024:
        st.warning(f"⚠️ 안테나 소자 수가 너무 많습니다 ({M} × {N} = {M*N}개).")
        st.info("웹 브라우저 성능 보호를 위해 1,024개 이상의 안테나 소자에 대한 개별 그래픽 플롯 출력을 제한합니다.")
        st.markdown(f"""
        **안테나 배열 주요 정보:**
        * **총 소자 수:** {M*N} 개
        * **수평 안테나(N):** {N} 개
        * **수직 안테나(M):** {M} 개
        * **안테나 간격:** {d_lambda} λ ({d*1000:.2f} mm)
        * **배열 가로 폭:** {(N-1)*d_lambda:.2f} λ
        * **배열 세로 높이:** {(M-1)*d_lambda:.2f} λ
        """)
    else:
        st.markdown("각 원은 개별 안테나 소자를 나타내며, 적용되는 **가중치 위상(Degree)**을 색상과 텍스트로 시각화합니다.")
        
        # 안테나 소자 데이터 매핑
        y_positions = []
        z_positions = []
        phases_list = []
        hover_labels = []
        
        for i in range(M):
            for j in range(N):
                y_pos = n_indices[j] * d_lambda
                z_pos = m_indices[i] * d_lambda
                p_deg = phases_deg[i, j]
                
                y_positions.append(y_pos)
                z_positions.append(z_pos)
                phases_list.append(p_deg)
                hover_labels.append(f"소자 ({i+1}, {j+1})<br>위상: {p_deg:.1f}°<br>위치: (y={y_pos:.2f}λ, z={z_pos:.2f}λ)")
                
        fig_ant = go.Figure()
        
        # Plotly Scatter 차트를 이용해 UPA 형상 가시화
        fig_ant.add_trace(go.Scatter(
            x=y_positions,
            y=z_positions,
            mode='markers+text',
            marker=dict(
                size=36,
                color=phases_list,
                colorscale='Hsv',  # 0~360도 표현용 순환형 컬러맵
                cmin=0,
                cmax=360,
                showscale=True,
                colorbar=dict(
                    title="위상 (°)",
                    tickvals=[0, 90, 180, 270, 360],
                    thickness=15
                ),
                line=dict(width=1, color='black')
            ),
            text=[f"{p:.0f}°" for p in phases_list],
            textposition="middle center",
            textfont=dict(color='white', size=10, family="Arial Black"),
            hoverinfo='text',
            hovertext=hover_labels
        ))
        
        # 축 격자 및 1:1 종횡비 설정
        fig_ant.update_layout(
            xaxis=dict(
                title="수평 방향 (Horizontal, λ)",
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='gray',
                zerolinewidth=1,
                range=[-N*d_lambda/2 - 0.5, N*d_lambda/2 + 0.5],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                title="수직 방향 (Vertical, λ)",
                gridcolor='rgba(128,128,128,0.2)',
                zerolinecolor='gray',
                zerolinewidth=1,
                range=[-M*d_lambda/2 - 0.5, M*d_lambda/2 + 0.5]
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            width=550,
            height=550,
            showlegend=False,
            margin=dict(l=10, r=10, b=10, t=40)
        )
        
        st.plotly_chart(fig_ant, use_container_width=True)

# 자동 스캔 애니메이션 갱신을 위한 루프 제어
if st.session_state.is_scanning:
    import time
    time.sleep(scan_delay)
    total_steps = az_steps * el_steps
    if st.session_state.scan_idx < total_steps - 1:
        st.session_state.scan_idx += 1
        st.rerun()
    else:
        st.session_state.is_scanning = False
        st.success("🔄 모든 영역 스캔 완료!")
        st.rerun()