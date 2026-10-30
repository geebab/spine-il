"""Joy-Con (L) → SO-101 end-effector teleoperation through the arm server (owner's reference client).

이 스크립트는 시리얼 포트를 열지 않는다. `spine-arm-server`(arm-server@left/right)에 리스를 잡고 관절 목표(°)를
30 Hz로 보낸다 — 급발진 방지 토크 인가·관절 경계·틱당 상한·워치독·과부하 감지·2단계 중립 복귀는 전부 서버 안에 있다.
IK·작업영역 상자·마커 프레임은 클라이언트 라이브러리 `lerobot_robot_spinearm.ee`(공개 레포)에 있다.

Wayland에서는 LeRobot의 키보드 텔레옵이 동작하지 않고(전역 키 후킹 불가), gamepad 텔레옵은 SDL이
단일 Joy-Con을 조이스틱으로 잡지 못한다. 대신 커널이 이미 노출한 evdev 장치를 직접 읽는다(`input` 그룹 필요 → owner1 전용).

조작 — Joy-Con (L)을 **옆으로 눕혀 한 손으로 잡은 자세(solo grip)** 기준.
이 자세에서는 십자·스틱이 90° 돌아가므로 커널 이름과 체감 방향이 다르다(RIGHT=위, DOWN=오른쪽).
  데드맨          누르고 있는 동안에만 움직인다(기본 BTN_TL, `--deadman`으로 변경)
  ZL (BTN_TL2)    누르고 있는 동안 **손목 직접 조그**: 스틱 위아래=wrist_flex, 좌우=wrist_roll
  스틱            위=마커 앞(+x) / 오른쪽=마커 오른쪽(−y)
  십자 위·아래     마커 z 위/아래           십자 왼·오른쪽   **6번 관절(마커 회전) 조그**
  − (SELECT)      종료                   `--grip vertical`  세워 잡는 자세 매핑

종료(−): 서버의 `home`(6번 먼저 접고 나머지) → 중립이면 `torque_off` → 리스 반납. 복귀가 막히면 토크를 켠 채 둔다.
Ctrl+C는 비상용 — 그 자리에서 정지(토크 유지, 리스 반납). 언제든 다른 셸에서 `spine-arm-ctl --arm left estop`.

  python robot/joycon_ee_teleop.py --dry-run                 # 로봇 없이 Joy-Con 매핑만 확인
  python robot/joycon_ee_teleop.py --arm left                # 실물(arm-server@left)
  python robot/joycon_ee_teleop.py --arm left --cmd-port 6601   # 가상 팔(spine-arm-server --sim --cmd-port 6601)
"""

from __future__ import annotations

import argparse
import getpass
import shutil
import threading
import time

import evdev

from lerobot_robot_spinearm.protocol import JOINTS

RATE_HZ = 30
DEADZONE = 0.12


class JoyConL:
    """evdev로 Joy-Con (L)을 읽는 최소 리더(백그라운드 스레드)."""

    # solo(옆으로 눕혀 잡기): 커널 십자 이름 → 체감 방향. vertical: 세워 잡기(그대로).
    DPAD = {
        "solo": {"up": "BTN_DPAD_RIGHT", "down": "BTN_DPAD_LEFT", "left": "BTN_DPAD_UP", "right": "BTN_DPAD_DOWN"},
        "vertical": {"up": "BTN_DPAD_UP", "down": "BTN_DPAD_DOWN", "left": "BTN_DPAD_LEFT", "right": "BTN_DPAD_RIGHT"},
    }

    def __init__(self, grip: str = "solo", deadman: str = "BTN_TL"):
        self.dpad = self.DPAD[grip]
        self.grip = grip
        self.deadman = deadman
        self.dev = next(
            (evdev.InputDevice(p) for p in evdev.list_devices() if "Joy-Con (L)" == evdev.InputDevice(p).name), None
        )
        if self.dev is None:
            raise SystemExit("Joy-Con (L)을 찾지 못했습니다 — 블루투스 연결 확인")
        self.stick = {"x": 0.0, "y": 0.0}
        self.keys: set[str] = set()
        self.quit = False
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for ev in self.dev.read_loop():
            if ev.type == evdev.ecodes.EV_ABS:
                axis = "x" if ev.code == evdev.ecodes.ABS_X else "y" if ev.code == evdev.ecodes.ABS_Y else None
                if axis:
                    v = ev.value / 32767.0
                    self.stick[axis] = 0.0 if abs(v) < DEADZONE else v
            elif ev.type == evdev.ecodes.EV_KEY:
                name = evdev.ecodes.KEY.get(ev.code) or evdev.ecodes.BTN.get(ev.code)
                name = name[0] if isinstance(name, list) else name
                if ev.value:
                    self.keys.add(name)
                    if name == "BTN_SELECT":
                        self.quit = True
                else:
                    self.keys.discard(name)

    def felt_stick(self) -> tuple[float, float]:
        """체감 (위+, 오른쪽+). solo 자세에서는 커널 X=체감 위, 커널 Y=체감 오른쪽."""
        if self.grip == "solo":
            return self.stick["x"], self.stick["y"]
        return -self.stick["y"], self.stick["x"]

    def jog(self) -> dict[str, float]:
        """ZL을 누르고 있는 동안 손목을 직접 움직인다(−1~1, 크기는 --jog-deg가 결정)."""
        if "BTN_TL2" not in self.keys:
            return {}
        up, right = self.felt_stick()
        return {"wrist_flex": up, "wrist_roll": right}

    def command(self) -> dict:
        """EEReferenceAndDelta가 받는 액션 키로 변환(방향은 −1/0/+1, 크기는 step_sizes가 결정)."""
        held = self.deadman in self.keys
        up, right = self.felt_stick()
        return {
            "enabled": held,
            "target_x": up if held else 0.0,  # 스틱 위 = 팁 앞(+x)
            "target_y": -right if held else 0.0,  # 스틱 오른쪽 = 팁 오른쪽(−y)
            "target_z": (1.0 if self.dpad["up"] in self.keys else -1.0 if self.dpad["down"] in self.keys else 0.0)
            if held
            else 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            # 6번 관절(=옛 그리퍼). 조(jaw)를 떼고 마커를 달았으므로 '축'으로 쓴다. 데드맨 게이트 안.
            "j6_vel": (1.0 if self.dpad["right"] in self.keys else -1.0 if self.dpad["left"] in self.keys else 0.0)
            if held
            else 0.0,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int, default=None, help="서버 명령 포트(기본 5601/5611; 가상 팔이면 6601 등)")
    ap.add_argument("--owner", default=getpass.getuser(), help="리스 이름")
    ap.add_argument("--step-m", type=float, default=0.006, help="스틱을 끝까지 밀었을 때 틱당 이동(m) — 0.006 ≈ 18 cm/s")
    ap.add_argument("--jog-deg", type=float, default=1.0, help="손목·6번 조그의 틱당 각도(°) — 1.0 ≈ 30°/s")
    ap.add_argument("--max-relative-target", type=float, default=10.0, help="관절 틱당 변화 상한(°) — 서버 step_deg")
    ap.add_argument("--no-return", action="store_true", help="종료 시 복귀하지 않고 그 자리에서 토크 유지")
    ap.add_argument("--orientation-weight", type=float, default=0.01,
                    help="IK 마커 방향 가중치(rad당, 위치는 m당 1). 0.01 = 1 rad ≈ 1 cm — 6관절 모두 자세 유지에 참여")
    ap.add_argument("--grip", choices=["solo", "vertical"], default="solo", help="Joy-Con 잡는 자세")
    ap.add_argument("--deadman", default="BTN_TL", help="데드맨 버튼(dry-run의 keys= 출력에서 이름 확인)")
    ap.add_argument("--dry-run", action="store_true", help="로봇 없이 Joy-Con 입력만 표시")
    args = ap.parse_args()

    joy = JoyConL(grip=args.grip, deadman=args.deadman)
    print(f"Joy-Con: {joy.dev.name} ({joy.dev.path})")
    if args.dry_run:
        print(f"데드맨({args.deadman})을 누른 채 움직여 보세요. 다른 버튼을 쓰려면 keys= 에 찍힌 이름을 --deadman 으로. − 종료.")
        while not joy.quit:
            c = joy.command()
            print(f"\r enabled={c['enabled']}  x{c['target_x']:+.2f} y{c['target_y']:+.2f} z{c['target_z']:+.0f}"
                  f"  j6{c['j6_vel']:+.0f}  keys={sorted(joy.keys)}          ", end="", flush=True)
            time.sleep(0.05)
        print()
        return 0

    from lerobot_robot_spinearm import ArmError, SpineArm, SpineArmConfig
    from lerobot_robot_spinearm.ee import TARGET_FRAME, EETeleop

    arm = SpineArm(SpineArmConfig(arm=args.arm, host=args.host, cmd_port=args.cmd_port, owner=args.owner,
                                  step_deg=args.max_relative_target))
    arm.connect()  # 리스 + (꺼져 있으면) 급발진 없는 토크 인가 — 순서는 서버가 지킨다
    st = arm.status()
    tele = EETeleop(step_m=args.step_m, jog_deg=args.jog_deg, orientation_weight=args.orientation_weight)
    if st.get("bounds"):
        tele.kin.set_limits(st["bounds"])  # IK joint limits = this arm's calibration (server clamps anyway)
    print(f"팔 {args.arm}: 상태 {st['state']} · 서버 {arm.rpc.endpoint} · 중립 {'있음' if st.get('neutral') else '없음(복귀 불가)'}"
          f"{' · 가상 팔' if st.get('sim') else ''}")
    print(f"마커 프레임 {TARGET_FRAME} · 6축 IK · 작업영역(m): x {tele.lo[0]:.2f}~{tele.hi[0]:.2f}"
          f" · y {tele.lo[1]:.2f}~{tele.hi[1]:.2f} · z {tele.lo[2]:.2f}~{tele.hi[2]:.2f}")
    print(f"{args.deadman}: 스틱=x/y · 십자 위아래=z · 십자 좌우=6번(마커 회전) | ZL: 스틱 위아래=wrist_flex, 좌우=wrist_roll | − 종료")
    last_renew, last_print, last_err = time.monotonic(), 0.0, ""
    try:
        while not joy.quit:
            t0 = time.perf_counter()
            obs = arm.get_observation()
            q = {j: obs[f"{j}.pos"] for j in JOINTS}
            jog = joy.jog()
            cmd = joy.command()
            mode = "■"
            try:
                if jog:  # ZL: 손목 조그 — 그 관절을 직접 돌리고 마커 목표 자세를 그 결과로 다시 잡는다
                    if any(jog.values()):
                        arm.send_action({f"{j}.pos": v for j, v in tele.jog(q, jog).items()})
                    mode = "✱"
                elif cmd["enabled"]:
                    goals = tele.step(q, (cmd["target_x"], cmd["target_y"], cmd["target_z"]), {"gripper": cmd["j6_vel"]})
                    arm.send_action({f"{j}.pos": v for j, v in goals.items()})
                    mode = "▶"
                else:
                    # 유지: 아무것도 보내지 않는다(서버가 250 ms 뒤 HOLD). 리스만 살려 둔다
                    tele.reset()
                    if time.monotonic() - last_renew > 5.0:
                        arm.renew()
                        last_renew = time.monotonic()
                last_err = ""
            except ArmError as e:
                if e.code in ("bad_token", "lease_held"):
                    print(f"\n리스를 잃었습니다({e}) — 종료합니다(팔은 서버가 그 자리에 유지)")
                    return 1
                if str(e) != last_err:  # fault / estop / bounds…: 한 번만 알리고 계속 관찰
                    print(f"\n서버 거부: {e}   (해결: spine-arm-ctl --arm {args.arm} clear | estop 후 lease)")
                    last_err = str(e)
                tele.reset()
            if time.monotonic() - last_print > 0.1:  # 10 Hz, one line: gnome-terminal crashed under a 30 Hz flood
                snap = arm.last or {}
                line = (f"{mode} {snap.get('state', '?'):7s} " + " ".join(f"{j[:5]}{v:7.1f}" for j, v in q.items()) +
                        (f"  ! {','.join(snap['faults'])}" if snap.get("faults") else ""))
                cols = shutil.get_terminal_size((100, 20)).columns - 1
                print("\r" + line[:cols].ljust(cols), end="", flush=True)
                last_print = time.monotonic()
            time.sleep(max(0.0, 1 / RATE_HZ - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if joy.quit and not args.no_return:
            print("\n복귀 중(6번 먼저 접고 나머지)…", flush=True)
            try:
                s = arm.home(wait=True)
                if s["state"] == "HOLD":
                    arm.torque_off()
                    print("복귀 완료 — 토크 해제")
                else:
                    print(f"복귀 미완({s['state']} {s.get('faults')}) — **토크를 켠 채 둡니다**(팔 낙하 방지).")
                    print(f"  걸림 제거 후: spine-arm-ctl --arm {args.arm} lease && spine-arm-ctl --arm {args.arm} clear && "
                          f"spine-arm-ctl --arm {args.arm} home --wait && spine-arm-ctl --arm {args.arm} torque-off")
            except KeyboardInterrupt:
                arm.hold()
                print("\n복귀 중단 — 그 자리에서 정지(토크 유지)")
            except ArmError as e:
                print(f"복귀 실패: {e} — 토크 유지")
        else:
            print(f"\n정지(토크 유지). 힘을 빼려면: spine-arm-ctl --arm {args.arm} lease && spine-arm-ctl --arm {args.arm} home --wait"
                  f" && spine-arm-ctl --arm {args.arm} torque-off")
        arm.disconnect()  # hold + 리스 반납(토크는 유지)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
