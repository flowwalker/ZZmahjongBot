"""国标麻将 Botzone Agent — 协议解析与状态管理."""

class MahjongGBAgent:
    """国标麻将 Agent 基类"""

    observation_space = None
    action_space = None

    def __init__(self, seatWind: int):
        """
        初始化 Agent

        Args:
            seatWind: 玩家的门风位置 (0=东, 1=南, 2=西, 3=北)
        """
        pass

    def request2obs(self, request: str):
        """
        将 Botzone 的 request 字符串解析为内部状态更新，
        并在需要决策时返回观测 (observation + action_mask)。

        Args:
            request: Botzone 格式的请求字符串

        Returns:
            dict or None: 如果需要决策，返回 {'observation': np.ndarray, 'action_mask': np.ndarray}
                          如果仅更新状态，返回 None
        """
        pass

    def action2response(self, action: int) -> str:
        """
        将整数动作编码转换为 Botzone response 字符串。

        Args:
            action: 动作整数编码 (0-234)

        Returns:
            Botzone 格式的响应字符串，如 "Play W3", "Hu", "Pass" 等
        """
        pass
