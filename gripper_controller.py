#%%
import requests
from typing import Optional

class GripperController:
    """夹爪控制器类"""
    
    def __init__(self, base_url: str = "http://192.168.40.102:8000"):
        """
        初始化夹爪控制器
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url.rstrip('/')
        
    def control(self, action: str, force: Optional[float] = 0.0) -> dict:
        """
        控制夹爪动作
        
        Args:
            action: 动作类型 ('open', 'close', 'switch')
            force: 夹持力度 (仅在close动作时有效)
            
        Returns:
            dict: API响应结果
            
        Raises:
            requests.exceptions.RequestException: 请求失败时抛出异常
        """
        url = f"{self.base_url}/gripper/control"
        payload = {
            "action": action,
            "force": force
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()  # 如果请求失败则抛出异常
        return response.json()
    
    def open(self) -> dict:
        """打开夹爪"""
        return self.control("open")
    
    def close(self, force: float = 100.0) -> dict:
        """关闭夹爪"""
        return self.control("close", force=force)
    
    def switch(self) -> dict:
        """切换夹爪状态"""
        return self.control("switch")

# %%
if __name__ == "__main__":
    gripper = GripperController()
    try:
        # 打开夹爪
        print("Opening gripper...")
        result = gripper.open()
        print(f"Result: {result}")
        
        # 关闭夹爪
        print("\nClosing gripper...")
        result = gripper.close(force=50.0)
        print(f"Result: {result}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")