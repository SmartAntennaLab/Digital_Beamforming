# 공개 서버 배포 보호

`nginx.conf.example`은 TLS, Basic Auth, IP별 HTTP 요청 제한, 연결 수 제한과
Streamlit WebSocket 프록시 헤더를 포함합니다. 실제 도메인·인증서 경로를 바꾸고
비밀번호 파일은 저장소 밖에서 생성하십시오.

```bash
sudo htpasswd -c /etc/nginx/dbf.htpasswd operator
sudo nginx -t
sudo systemctl reload nginx
```

Nginx의 요청 제한은 WebSocket 연결 이후의 개별 Streamlit 계산 이벤트를 세지
못합니다. 앱 내부의 세션 token bucket과 함께 사용해야 합니다. 여러 Streamlit
프로세스나 replica를 운영하면 앱 semaphore도 프로세스별로 존재하므로, proxy나
배포 플랫폼에서 전체 replica 수·CPU quota·메모리 quota를 추가로 제한하십시오.

`digital-beamforming.service.example`은 앱을 loopback에만 bind하고 systemd의
`CPUQuota`, `MemoryMax`, `TasksMax`를 적용하는 예입니다. 경로와 전용 사용자,
용량은 서버 사양에 맞게 수정하십시오. 운영 환경 변수는 권한을 제한한
`/etc/digital-beamforming.env`에 둡니다.

```dotenv
DBF_MAX_CONCURRENT_CALCULATIONS=2
DBF_COMPUTE_QUEUE_TIMEOUT_SECONDS=1
DBF_COMPUTE_TIMEOUT_SECONDS=10
DBF_SESSION_CALCULATIONS_PER_MINUTE=120
DBF_SESSION_BURST=8
DBF_HEALTH_LOG_INTERVAL_SECONDS=30
```

앱은 `compute_health {JSON}` 형식의 구조화 로그를 주기적으로 남깁니다. JSON에는
활성·대기 계산, 완료·혼잡·빈도 제한·시간 초과·취소 수, 평균 계산 시간, 프로세스
CPU/RSS와 시스템 CPU/메모리 사용률이 포함됩니다. 이 로그를 journald, Loki,
CloudWatch 등의 수집기로 전달하고 다음 항목에 경보를 구성하십시오.

- 활성 계산이 지속적으로 semaphore 상한에 도달
- 혼잡 또는 빈도 제한 거절이 5분 이상 증가
- 시간 초과가 반복 발생
- 시스템 CPU 90% 이상 또는 메모리 85% 이상 지속
- systemd의 OOM 종료나 재시작 횟수 증가
