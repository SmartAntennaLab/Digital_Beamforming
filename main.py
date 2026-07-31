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

st.title("📡 디지털 빔포밍 시뮬레이터 v1.1")
st.markdown("주파수, 안테나 배열, 조향 각도(Azimuth/Elevation)를 조절하여 빔 패턴과 안테나 상태를 확인하세요.")

# --- 사이드바: 입력 파라미터 ---
st.sidebar.header("⚙️ 시뮬레이터 입력 설정")

freq_ghz = st.sidebar.slider("주파수 (GHz)", min_value=1.0, max_value=60.0, value=28.0, step=0.5)
M = st.sidebar.slider("수직 안테나 수 (M)", min_value=1, max_value=128, value=4, step=1)
N = st.sidebar.slider("수평 안테나 수 (N)", min_value=1, max_value=128, value=4, step=1)
d_lambda = st.sidebar.slider("안테나 간격 (파장 기준, d/λ)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
taper_option = st.sidebar.selectbox(
    "진폭 테이퍼링 (Window)",
    options=["Uniform (균일)", "Hamming", "Hanning", "Blackman", "Bartlett"],
    index=0
)
element_option = st.sidebar.selectbox(
    "안테나 소자 패턴 (Element Factor)",
    options=["Isotropic (등방성)", "Cosine (코사인)", "Cosine² (코사인 제곱)", "Dipole (다이폴)"],
    index=0
)

azimuth_deg = st.sidebar.slider("목표 Azimuth 각도 (°)", min_value=-90.0, max_value=90.0, value=0.0, step=1.0)
elevation_deg = st.sidebar.slider("목표 Elevation 각도 (°)", min_value=-90.0, max_value=90.0, value=0.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("📊 시각화 설정")
scale_option = st.sidebar.radio(
    "3D 빔 패턴 스케일", 
    options=["dB Scale (추천)", "Linear Scale"], 
    index=0
)
coord_option = st.sidebar.radio(
    "2D 패턴 좌표계",
    options=["Polar (극좌표)", "Rectangular (직각좌표)"],
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
        if st.button("▶️ 스캔 시작", disabled=st.session_state.is_scanning, width="stretch"):
            st.session_state.is_scanning = True
            st.session_state.scan_idx = 0
            st.rerun()
    with col_btn2:
        if st.button("⏹️ 스캔 중지", disabled=not st.session_state.is_scanning, width="stretch"):
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

# 진폭 가중치 계산 함수
def get_window_weights(length, option):
    if length <= 1:
        return np.ones(length)
    if option == "Hamming":
        w = np.hamming(length)
    elif option == "Hanning":
        w = np.hanning(length)
    elif option == "Blackman":
        w = np.blackman(length)
    elif option == "Bartlett":
        w = np.bartlett(length)
    else:
        w = np.ones(length)
    return w / np.max(w)

w_n = get_window_weights(N, taper_option)
w_m = get_window_weights(M, taper_option)

# 수평(Y축), 수직(Z축) 조향 위상 변화율 계산 (3D 빔 패턴과 물리 좌표 매치)
k_y = 2 * np.pi / lam * np.cos(el) * np.sin(az)  # 수평 방향 조향 성분
k_z = 2 * np.pi / lam * np.sin(el)               # 수직 방향 조향 성분

# 각 안테나 소자의 위치 및 가중치(위상) 계산
Y, Z = np.meshgrid(n_indices * d, m_indices * d)
phases = - (k_y * Y + k_z * Z)
phases_deg = np.degrees(phases) % 360  # 0~360 도로 표현
# --- 격자 로브 (Grating Lobe) 감지 ---
gl_az_angles = []
gl_el_angles = []

# k = -1, 1 (또는 더 큰 범위 체크)
for k in [-2, -1, 1, 2]:
    sin_val_az = np.sin(az) + k / d_lambda
    if -1.0001 <= sin_val_az <= 1.0001:
        gl_angle = np.degrees(np.arcsin(np.clip(sin_val_az, -1.0, 1.0)))
        gl_az_angles.append((k, gl_angle))
        
    sin_val_el = np.sin(el) + k / d_lambda
    if -1.0001 <= sin_val_el <= 1.0001:
        gl_angle = np.degrees(np.arcsin(np.clip(sin_val_el, -1.0, 1.0)))
        gl_el_angles.append((k, gl_angle))

if gl_az_angles or gl_el_angles:
    warning_msg = "⚠️ **격자 로브(Grating Lobe) 감지 경고**\n\n"
    warning_msg += "안테나 간격($d/\\lambda$)이 넓거나 조향 각도가 커서 가시 영역($-90^\\circ \\sim 90^\\circ$) 내에 원치 않는 강한 부엽(격자 로브)이 감지되었습니다. 이는 메인 빔과 동일한 세기로 다른 방향에 에너지가 방사되어 성능 저하 및 혼선을 유발합니다.\n\n"
    if gl_az_angles:
        warning_msg += "**수평(Azimuth) 격자 로브 위치:**\n"
        for k, angle in gl_az_angles:
            warning_msg += f"- 차수 $k={k}$ : `{angle:.1f}°`\n"
    if gl_el_angles:
        warning_msg += "\n**수직(Elevation) 격자 로브 위치:**\n"
        for m, angle in gl_el_angles:
            warning_msg += f"- 차수 $m={m}$ : `{angle:.1f}°`\n"
    warning_msg += "\n💡 **해결 방법**: 안테나 간격($d/\\lambda$)을 줄이거나(일반적으로 0.5 이하 권장) 조향 범위를 좁히십시오."
    st.warning(warning_msg)

# --- 탭 구성 ---
tab1, tab2, tab3 = st.tabs(["📊 빔 패턴 (2D/3D)", "🔍 3dB 대역폭", "🔴 안테나 배치 및 위상"])

with tab1:
    st.subheader("방사 패턴 (Radiation Pattern)")
    
    col1, col2 = st.columns(2)
    
    # 2D 스캔 범위 정의
    phi_scan = np.linspace(-np.pi/2, np.pi/2, 360)
    theta_scan = np.linspace(-np.pi/2, np.pi/2, 360)

    # 2D Element Factor 계산
    if element_option == "Cosine (코사인)":
        ef_2d_az = np.maximum(0.0, np.cos(phi_scan))
        ef_2d_el = np.maximum(0.0, np.cos(theta_scan))
    elif element_option == "Cosine² (코사인 제곱)":
        ef_2d_az = np.maximum(0.0, np.cos(phi_scan))**2
        ef_2d_el = np.maximum(0.0, np.cos(theta_scan))**2
    elif element_option == "Dipole (다이폴)":
        ef_2d_az = np.ones_like(phi_scan)
        ef_2d_el = np.cos((np.pi / 2.0) * np.sin(theta_scan)) / (np.cos(theta_scan) + 1e-6)
    else: # Isotropic
        ef_2d_az = np.ones_like(phi_scan)
        ef_2d_el = np.ones_like(theta_scan)

    # 2D Azimuth 컷 패턴 계산 (NumPy 벡터화 적용)
    # phi_scan: (360, 1), n_indices: (1, N) 형태로 브로드캐스팅
    steering_matrix = np.exp(1j * 2 * np.pi / lam * d * n_indices[None, :] * (np.sin(phi_scan[:, None]) - np.sin(az)))
    af_2d = np.sum(steering_matrix * w_n[None, :], axis=1) * ef_2d_az
    af_2d_db = 20 * np.log10(np.abs(af_2d) / np.max(np.abs(af_2d)) + 1e-6)
    
    # 2D Elevation 컷 패턴 계산 (NumPy 벡터화 적용)
    # theta_scan: (360, 1), m_indices: (1, M) 형태로 브로드캐스팅
    steering_matrix_el = np.exp(1j * 2 * np.pi / lam * d * m_indices[None, :] * (np.sin(theta_scan[:, None]) - np.sin(el)))
    af_2d_el = np.sum(steering_matrix_el * w_m[None, :], axis=1) * ef_2d_el
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
        st.markdown(f"#### 2D Azimuth 빔 패턴 ({'Polar' if coord_option == 'Polar (극좌표)' else 'Rectangular'})")
        fig_az = go.Figure()
        
        phi_scan_deg = np.degrees(phi_scan)
        
        if coord_option == "Polar (극좌표)":
            # Main trace
            fig_az.add_trace(go.Scatterpolar(
                r=af_2d_db,
                theta=phi_scan_deg,
                mode='lines',
                line=dict(color='dodgerblue', width=2),
                name='Gain (dB)',
                hovertemplate='Angle: %{theta:.2f}°<br>Gain: %{r:.2f} dB<extra></extra>'
            ))
            
            # -3dB line
            fig_az.add_trace(go.Scatterpolar(
                r=[-3]*len(phi_scan_deg),
                theta=phi_scan_deg,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # 3dB Range Fill
            if show_3db and idx_left_az is not None and idx_right_az is not None:
                fill_theta = list(phi_scan_deg[idx_left_az:idx_right_az+1])
                fill_r = list(af_2d_db[idx_left_az:idx_right_az+1])
                # Close trace to the center/bottom limit (-40 dB)
                fill_theta += [fill_theta[-1], fill_theta[0]]
                fill_r += [-40, -40]
                
                lbl = f'3dB BW: {bw_deg_az:.2f}°' if show_3db_value else '3dB Beamwidth'
                fig_az.add_trace(go.Scatterpolar(
                    r=fill_r,
                    theta=fill_theta,
                    fill='toself',
                    fillcolor='rgba(30, 144, 255, 0.15)',
                    line=dict(color='rgba(30, 144, 255, 0.5)', width=1, dash='dash'),
                    name=lbl,
                    hoverinfo='skip'
                ))
            elif not show_3db and show_3db_value:
                # Add a dummy trace for legend info if needed
                if idx_left_az is not None and idx_right_az is not None:
                    fig_az.add_trace(go.Scatterpolar(
                        r=[None], theta=[None],
                        mode='markers',
                        marker=dict(color='dodgerblue', opacity=0),
                        name=f'3dB BW: {bw_deg_az:.2f}°'
                    ))
            
            fig_az.update_layout(
                polar=dict(
                    angularaxis=dict(direction="clockwise", rotation=90, ticksuffix="°"),
                    radialaxis=dict(range=[-40, 0], tickvals=[-40, -30, -20, -10, -3, 0], ticksuffix=" dB")
                ),
                margin=dict(l=30, r=30, t=40, b=30),
                showlegend=(show_3db or show_3db_value),
                height=450,
                legend=dict(yanchor="top", y=-0.1, xanchor="center", x=0.5, orientation="h")
            )
            
        else: # Rectangular
            fig_az.add_trace(go.Scatter(
                x=phi_scan_deg,
                y=af_2d_db,
                mode='lines',
                line=dict(color='dodgerblue', width=2),
                name='Gain (dB)',
                hovertemplate='Angle: %{x:.2f}°<br>Gain: %{y:.2f} dB<extra></extra>'
            ))
            
            fig_az.add_trace(go.Scatter(
                x=[-90, 90],
                y=[-3, -3],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            if show_3db and idx_left_az is not None and idx_right_az is not None:
                fill_x = list(phi_scan_deg[idx_left_az:idx_right_az+1])
                fill_y = list(af_2d_db[idx_left_az:idx_right_az+1])
                fill_x = [fill_x[0]] + fill_x + [fill_x[-1]]
                fill_y = [-40] + fill_y + [-40]
                
                lbl = f'3dB BW: {bw_deg_az:.2f}°' if show_3db_value else '3dB Beamwidth'
                fig_az.add_trace(go.Scatter(
                    x=fill_x,
                    y=fill_y,
                    fill='toself',
                    fillcolor='rgba(30, 144, 255, 0.15)',
                    line=dict(color='rgba(30, 144, 255, 0.5)', width=1, dash='dash'),
                    name=lbl,
                    hoverinfo='skip'
                ))
            elif not show_3db and show_3db_value:
                if idx_left_az is not None and idx_right_az is not None:
                    fig_az.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(color='dodgerblue', opacity=0),
                        name=f'3dB BW: {bw_deg_az:.2f}°'
                    ))
            
            fig_az.update_layout(
                xaxis=dict(title="Angle (°)", range=[-90, 90], gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(title="Normalized Gain (dB)", range=[-40, 0], gridcolor='rgba(128,128,128,0.2)'),
                margin=dict(l=30, r=30, t=40, b=30),
                showlegend=(show_3db or show_3db_value),
                height=450,
                legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5, orientation="h")
            )
            
        st.plotly_chart(fig_az, width="stretch")

    with col2:
        st.markdown(f"#### 2D Elevation 빔 패턴 ({'Polar' if coord_option == 'Polar (극좌표)' else 'Rectangular'})")
        fig_el = go.Figure()
        
        theta_scan_deg = np.degrees(theta_scan)
        
        if coord_option == "Polar (극좌표)":
            fig_el.add_trace(go.Scatterpolar(
                r=af_2d_el_db,
                theta=theta_scan_deg,
                mode='lines',
                line=dict(color='crimson', width=2),
                name='Gain (dB)',
                hovertemplate='Angle: %{theta:.2f}°<br>Gain: %{r:.2f} dB<extra></extra>'
            ))
            
            fig_el.add_trace(go.Scatterpolar(
                r=[-3]*len(theta_scan_deg),
                theta=theta_scan_deg,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            if show_3db and idx_left_el is not None and idx_right_el is not None:
                fill_theta = list(theta_scan_deg[idx_left_el:idx_right_el+1])
                fill_r = list(af_2d_el_db[idx_left_el:idx_right_el+1])
                fill_theta += [fill_theta[-1], fill_theta[0]]
                fill_r += [-40, -40]
                
                lbl = f'3dB BW: {bw_deg_el:.2f}°' if show_3db_value else '3dB Beamwidth'
                fig_el.add_trace(go.Scatterpolar(
                    r=fill_r,
                    theta=fill_theta,
                    fill='toself',
                    fillcolor='rgba(220, 20, 60, 0.15)',
                    line=dict(color='rgba(220, 20, 60, 0.5)', width=1, dash='dash'),
                    name=lbl,
                    hoverinfo='skip'
                ))
            elif not show_3db and show_3db_value:
                if idx_left_el is not None and idx_right_el is not None:
                    fig_el.add_trace(go.Scatterpolar(
                        r=[None], theta=[None],
                        mode='markers',
                        marker=dict(color='crimson', opacity=0),
                        name=f'3dB BW: {bw_deg_el:.2f}°'
                    ))
            
            fig_el.update_layout(
                polar=dict(
                    angularaxis=dict(direction="clockwise", rotation=90, ticksuffix="°"),
                    radialaxis=dict(range=[-40, 0], tickvals=[-40, -30, -20, -10, -3, 0], ticksuffix=" dB")
                ),
                margin=dict(l=30, r=30, t=40, b=30),
                showlegend=(show_3db or show_3db_value),
                height=450,
                legend=dict(yanchor="top", y=-0.1, xanchor="center", x=0.5, orientation="h")
            )
            
        else: # Rectangular
            fig_el.add_trace(go.Scatter(
                x=theta_scan_deg,
                y=af_2d_el_db,
                mode='lines',
                line=dict(color='crimson', width=2),
                name='Gain (dB)',
                hovertemplate='Angle: %{x:.2f}°<br>Gain: %{y:.2f} dB<extra></extra>'
            ))
            
            fig_el.add_trace(go.Scatter(
                x=[-90, 90],
                y=[-3, -3],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            if show_3db and idx_left_el is not None and idx_right_el is not None:
                fill_x = list(theta_scan_deg[idx_left_el:idx_right_el+1])
                fill_y = list(af_2d_el_db[idx_left_el:idx_right_el+1])
                fill_x = [fill_x[0]] + fill_x + [fill_x[-1]]
                fill_y = [-40] + fill_y + [-40]
                
                lbl = f'3dB BW: {bw_deg_el:.2f}°' if show_3db_value else '3dB Beamwidth'
                fig_el.add_trace(go.Scatter(
                    x=fill_x,
                    y=fill_y,
                    fill='toself',
                    fillcolor='rgba(220, 20, 60, 0.15)',
                    line=dict(color='rgba(220, 20, 60, 0.5)', width=1, dash='dash'),
                    name=lbl,
                    hoverinfo='skip'
                ))
            elif not show_3db and show_3db_value:
                if idx_left_el is not None and idx_right_el is not None:
                    fig_el.add_trace(go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(color='crimson', opacity=0),
                        name=f'3dB BW: {bw_deg_el:.2f}°'
                    ))
            
            fig_el.update_layout(
                xaxis=dict(title="Angle (°)", range=[-90, 90], gridcolor='rgba(128,128,128,0.2)'),
                yaxis=dict(title="Normalized Gain (dB)", range=[-40, 0], gridcolor='rgba(128,128,128,0.2)'),
                margin=dict(l=30, r=30, t=40, b=30),
                showlegend=(show_3db or show_3db_value),
                height=450,
                legend=dict(yanchor="top", y=-0.2, xanchor="center", x=0.5, orientation="h")
            )
            
        st.plotly_chart(fig_el, width="stretch")

    st.markdown("---")
    st.markdown("#### 3D 빔 패턴 (Spherical Surface)")
    
    # 대형 배열에 따른 3D 해상도 동적 설정 (성능 최적화)
    total_elements = M * N
    if total_elements <= 256:
        grid_res = 50
    elif total_elements <= 1024:
        grid_res = 40
    elif total_elements <= 4096:
        grid_res = 30
    else:
        grid_res = 20

    # 3D 구면 좌표계 패턴 벡터화 계산
    theta_3d = np.linspace(0, np.pi, grid_res)  # 극각
    phi_3d = np.linspace(-np.pi, np.pi, grid_res)  # 방위각
    THETA, PHI = np.meshgrid(theta_3d, phi_3d)
    
    # 안테나 배열 소자들의 좌표를 그리드로 생성
    Y_pos, Z_pos = np.meshgrid(n_indices * d, m_indices * d)
    Y_flat = Y_pos.ravel()[None, None, :]  # shape: (1, 1, M*N)
    Z_flat = Z_pos.ravel()[None, None, :]  # shape: (1, 1, M*N)
    
    T_grid = THETA[:, :, None]             # shape: (50, 50, 1)
    P_grid = PHI[:, :, None]               # shape: (50, 50, 1)
    
    # Kronecker 외적으로 2D 윈도우 가중치 매트릭스 계산 후 플랫화
    W_2d = np.outer(w_m, w_n)
    W_flat = W_2d.ravel()[None, None, :]
    
    # 브로드캐스팅 연산을 통해 모든 안테나 소자의 phase shift 한 번에 계산
    phase_shift = 2 * np.pi / lam * (Y_flat * np.sin(T_grid) * np.sin(P_grid) + Z_flat * np.cos(T_grid))
    target_shift = 2 * np.pi / lam * (Y_flat * np.cos(el) * np.sin(az) + Z_flat * np.sin(el))
    
    AF_3D = np.sum(W_flat * np.exp(1j * (phase_shift - target_shift)), axis=2)

    # 3D Element Factor 계산
    if element_option == "Cosine (코사인)":
        cos_psi = np.sin(THETA) * np.cos(PHI)
        EF_3D = np.maximum(0.0, cos_psi)
    elif element_option == "Cosine² (코사인 제곱)":
        cos_psi = np.sin(THETA) * np.cos(PHI)
        EF_3D = np.maximum(0.0, cos_psi)**2
    elif element_option == "Dipole (다이폴)":
        EF_3D = np.cos((np.pi / 2.0) * np.cos(THETA)) / (np.sin(THETA) + 1e-6)
    else: # Isotropic
        EF_3D = np.ones_like(THETA)

    AF_3D = AF_3D * EF_3D
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
    st.plotly_chart(fig_3d, width="stretch")
    st.caption(f"⚡ 대형 배열 연산 최적화 적용됨 (3D 해상도: {grid_res}x{grid_res})")

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

    import pandas as pd
    
    # 2D 빔포밍 데이터 프레임 구축
    df_export = pd.DataFrame({
        "Angle (deg)": np.degrees(phi_scan),
        "Azimuth Gain (dB)": af_2d_db,
        "Elevation Gain (dB)": af_2d_el_db
    })
    csv_data = df_export.to_csv(index=False).encode('utf-8')
    
    # 마크다운 설계 리포트 작성
    report_content = f"""# 📡 디지털 빔포밍 안테나 설계 리포트 (Design Report)

본 문서는 디지털 빔포밍 시뮬레이터에서 연산된 안테나 구성 파라미터 및 성능 분석 결과를 요약한 보고서입니다.

---

## 1. 안테나 시뮬레이션 기본 파라미터 (Simulation Parameters)
* **주파수 (Frequency)**: {freq_ghz} GHz (물리적 파장 $\\lambda$: {lam*1000:.2f} mm)
* **안테나 크기 (Array Configuration)**: {M} (수직 소자 수) × {N} (수평 소자 수) = 총 {M*N} 개 소자
* **소자 간 간격 (Spacing)**: {d_lambda} $\\lambda$ (물리적 간격: {d*1000:.2f} mm)
* **목표 조향각 (Steering Direction)**: Azimuth = {current_azimuth:.1f}°, Elevation = {current_elevation:.1f}°
* **진폭 가중치 기법 (Amplitude Tapering)**: {taper_option}
* **소자 물리 패턴 (Element Radiation Pattern)**: {element_option}

---

## 2. 주요 계산 성능 지표 (Key Performance Metrics)
* **배열 지향성 이득 (Array Directivity Gain)**: {array_gain_db:.2f} dBi
* **3dB 대역폭 (Half Power Beamwidth)**:
  - Azimuth 컷: {bw_deg_az:.2f}°
  - Elevation 컷: {bw_deg_el:.2f}°
* **주요 영점 대역폭 (First Null Bandwidth)**:
  - Azimuth: {fnbw_deg_az:.2f}°
  - Elevation: {fnbw_deg_el:.2f}°
* **최대 부엽 레벨 (Sidelobe Level - SLL)**:
  - Azimuth: {f"{sll_db_az:.2f} dB (@ {sll_angle_az:.1f}°)" if sll_db_az > -90.0 else "N/A (부엽 없음)"}
  - Elevation: {f"{sll_db_el:.2f} dB (@ {sll_angle_el:.1f}°)" if sll_db_el > -90.0 else "N/A (부엽 없음)"}

---

## 3. 격자 로브 (Grating Lobe) 감지 상태
"""
    if gl_az_angles or gl_el_angles:
        report_content += "* **상태**: ⚠️ 격자 로브 감지됨 (가시 영역 내에 원치 않는 메인 로브 중복 출현)\\n"
        if gl_az_angles:
            report_content += "* **수평(Azimuth) 격자 로브 위치**:\\n"
            for k_ord, angle in gl_az_angles:
                report_content += f"  - 차수 k={k_ord}: {angle:.1f}°\\n"
        if gl_el_angles:
            report_content += "* **수직(Elevation) 격자 로브 위치**:\\n"
            for m_ord, angle in gl_el_angles:
                report_content += f"  - 차수 m={m_ord}: {angle:.1f}°\\n"
    else:
        report_content += "* **상태**: ✅ 양호 (가시 영역 내 격자 로브 없음)\\n"

    report_content += "\\n---  \\n*본 리포트는 Digital Beamforming Simulator에 의해 자동으로 생성되었습니다.*"
    report_bytes = report_content.encode('utf-8')
    
    st.markdown("---")
    st.subheader("💾 데이터 내보내기 (Export Data)")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📊 2D 빔 패턴 CSV 다운로드",
            data=csv_data,
            file_name="beam_pattern_data.csv",
            mime="text/csv",
            width="stretch"
        )
    with col_dl2:
        st.download_button(
            label="📄 설계 리포트 (Markdown) 다운로드",
            data=report_bytes,
            file_name="beamforming_design_report.md",
            mime="text/markdown",
            width="stretch"
        )

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
        marker_sizes = []
        
        # 2D Kronecker 가중치
        W_2d = np.outer(w_m, w_n)
        
        for i in range(M):
            for j in range(N):
                y_pos = n_indices[j] * d_lambda
                z_pos = m_indices[i] * d_lambda
                p_deg = phases_deg[i, j]
                w_val = W_2d[i, j]
                
                y_positions.append(y_pos)
                z_positions.append(z_pos)
                phases_list.append(p_deg)
                marker_sizes.append(16 + 20 * w_val)
                hover_labels.append(
                    f"소자 ({i+1}, {j+1})<br>"
                    f"위상: {p_deg:.1f}°<br>"
                    f"진폭 가중치: {w_val:.3f}<br>"
                    f"위치: (y={y_pos:.2f}λ, z={z_pos:.2f}λ)"
                )
                
        fig_ant = go.Figure()
        
        # Plotly Scatter 차트를 이용해 UPA 형상 가시화
        fig_ant.add_trace(go.Scatter(
            x=y_positions,
            y=z_positions,
            mode='markers+text',
            marker=dict(
                size=marker_sizes,
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
        
        st.plotly_chart(fig_ant, width="stretch")

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