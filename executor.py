import pyautogui
import time


class MoveExecutor:
    def __init__(self, geometry, anchor, scale_xy):
        """
        geometry: GridGeometry (in INTERNAL coordinate space)
        anchor: (x, y) top-left of capture on screen
        scale_xy: (sx, sy) from INTERNAL → actual pixels
        """
        self.g = geometry
        self.anchor = anchor
        self.sx, self.sy = scale_xy

        self.terminate = False
        self.last_mouse_position = None

    # -------------------------
    # Coordinate conversion
    # -------------------------
    def _to_screen(self, x, y):
        sx = self.anchor[0] + x * self.sx
        sy = self.anchor[1] + y * self.sy
        return int(round(sx)), int(round(sy))

    def cell_center(self, r, c):
        """
        Center of (r, c) in INTERNAL space → screen
        """
        x = self.g.anchor_x + c * self.g.h + self.g.hwidth / 2
        y = self.g.anchor_y + r * self.g.v + self.g.vwidth / 2
        return self._to_screen(x, y)

    # -------------------------
    # Interrupt (move mouse to stop)
    # -------------------------
    def _check_interrupt(self):
        if self.terminate:
            return True

        current = pyautogui.position()

        if self.last_mouse_position is None:
            self.last_mouse_position = current
            return False

        dx = abs(current[0] - self.last_mouse_position[0])
        dy = abs(current[1] - self.last_mouse_position[1])

        if dx > 5 or dy > 5:
            print("User interrupted execution")
            self.terminate = True
            return True

        return False

    # -------------------------
    # Execute one move
    # -------------------------
    def execute_move(self, move, duration=0.12):
        """
        Move is EXCLUSIVE: [r1:r2, c1:c2]
        """
        if self._check_interrupt():
            return

        r1, c1, r2, c2 = move.r1, move.c1, move.r2, move.c2

        # exclusive bounds → bottom-right is (r2-1, c2-1)
        p1 = self.cell_center(r1, c1)
        p2 = self.cell_center(r2 - 1, c2 - 1)

        pyautogui.moveTo(*p1, duration=0.08)
        self.last_mouse_position = pyautogui.position()

        if self._check_interrupt(): return
        pyautogui.mouseDown()
        time.sleep(0.03)

        pyautogui.moveTo(*p2, duration=duration)
        self.last_mouse_position = pyautogui.position()

        if self._check_interrupt(): return
        time.sleep(0.03)
        pyautogui.mouseUp()

    # -------------------------
    # Execute all moves
    # -------------------------
    def execute(self, moves):
        self.terminate = False
        self.last_mouse_position = pyautogui.position()

        for i, move in enumerate(moves):
            if self.terminate:
                print(f"Stopped at move {i}")
                break

            self.execute_move(move)
    
    def click_play_again(self):
        if self.terminate: return
        pos = self.cell_center(11, 5)
        if self._check_interrupt() or self.terminate: return 
        pyautogui.click(pos[0], pos[1])
        self.last_mouse_position = pyautogui.position()
    
    def click_restart(self):
        pos = self.cell_center(-2, 0)
        if self._check_interrupt() or self.terminate: return 
        pyautogui.click(pos[0], pos[1])
        self.last_mouse_position = pyautogui.position()
        time.sleep(0.5)

        pos = self.cell_center(8, 4)
        if self._check_interrupt() or self.terminate: return 
        pyautogui.click(pos[0], pos[1])
        self.last_mouse_position = pyautogui.position()
        time.sleep(0.5)

        pos = self.cell_center(9, 4)
        if self._check_interrupt() or self.terminate: return 
        pyautogui.click(pos[0], pos[1])
        self.last_mouse_position = pyautogui.position()
        time.sleep(1)

        pos = self.cell_center(18, 5)
        if self._check_interrupt() or self.terminate: return 
        pyautogui.click(pos[0], pos[1])
        self.last_mouse_position = pyautogui.position()
        time.sleep(3)

# testing code
if __name__ == '__main__':
    from capture import capture_game_window
    from models import GridGeometry
    from recognizer import Recognizer

    cap = capture_game_window()
    recognizer = Recognizer()
    geometry = GridGeometry.from_sqinfo(recognizer.sqinfo)
    executor = MoveExecutor(
        geometry,
        cap.anchor,
        scale_xy=(
            cap.owidth / float(cap.internal_width),
            cap.oheight / float(cap.internal_height),
        )
    )
    executor.click_restart()