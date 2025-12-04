import asyncio
import logging
import time
from turtle import pos
from typing import Optional, Dict, Any, Tuple
import pyautogui
import win32gui
import win32con
import win32api
import win32clipboard


logger = logging.getLogger(__name__)


class WindowsActions:
    """Windows-specific UI actions using Windows API and PyAutoGUI"""
    
    def __init__(self):
        pass

    def getscreen_size(self) -> Tuple[int, int]:
        """Get the size of the primary screen"""
        try:
            screen_width, screen_height = pyautogui.size()
            return screen_width, screen_height
        except Exception as e:
            logger.error(f"Error getting screen size: {e}")
            return 1920, 1080
        
    async def click(self, x: float, y: float, button: str = 'left') -> bool:
        """Click at specified coordinates"""
        if (0 <= x <= 1 and 0 <= y <= 1):
            # If the coordinates are normalized (0 to 1)
            pos_x = self.getscreen_size()[0] * x
            pos_y = self.getscreen_size()[1] * y
        else:
            # If the coordinates are normalized (0 to 1000)
            pos_x = self.getscreen_size()[0] * (x / 1000)
            pos_y = self.getscreen_size()[1] * (y / 1000)
        try:
            if button == 'left':
                pyautogui.click(pos_x, pos_y)
            elif button == 'right':
                pyautogui.rightClick(pos_x, pos_y)
            elif button == 'double':
                pyautogui.doubleClick(pos_x, pos_y)

            await asyncio.sleep(0.1)  # Small delay after click
            return True
        except Exception as e:
            logger.error(f"Error clicking at ({x}, {y}): {e}")
            return False

    async def drag(self, start_x: float, start_y: float, end_x: float, end_y: float, duration: float = 0.5) -> bool:
        """Drag from start to end position"""
        if (0<= start_x <= 1 and 0 <= start_y <= 1 and 0 <= end_x <= 1 and 0 <= end_y <= 1):
            # If the coordinates are normalized (0 to 1)
            pos_start_x = self.getscreen_size()[0] * start_x
            pos_start_y = self.getscreen_size()[1] * start_y
            pos_end_x = self.getscreen_size()[0] * end_x
            pos_end_y = self.getscreen_size()[1] * end_y
        else:
            # If the coordinates are normalized (0 to 1000)
            pos_start_x = self.getscreen_size()[0] * (start_x / 1000)
            pos_start_y = self.getscreen_size()[1] * (start_y / 1000)
            pos_end_x = self.getscreen_size()[0] * (end_x / 1000)
            pos_end_y = self.getscreen_size()[1] * (end_y / 1000)
        try:
            pyautogui.drag(pos_end_x - pos_start_x, pos_end_y - pos_start_y, duration=duration, button='left')
            return True
        except Exception as e:
            logger.error(f"Error dragging from ({start_x}, {start_y}) to ({end_x}, {end_y}): {e}")
            return False

    async def scroll(self, x: float, y: float, clicks: int) -> bool:
        """Scroll at specified position"""
        if (0 <= x <= 1 and 0 <= y <= 1):
            # If the coordinates are normalized (0 to 1)
            pos_x = self.getscreen_size()[0] * x
            pos_y = self.getscreen_size()[1] * y
        else:
            # If the coordinates are normalized (0 to 1000)
            pos_x = self.getscreen_size()[0] * (x / 1000)
            pos_y = self.getscreen_size()[1] * (y / 1000)
        try:
            pyautogui.scroll(clicks, x=pos_x, y=pos_y)
            return True
        except Exception as e:
            logger.error(f"Error scrolling at ({x}, {y}): {e}")
            return False
    
    # async def type_text(self, text: str) -> bool:
    #     """
    #     Type text using clipboard method to handle both English and Chinese characters
    #     This method bypasses input method editors (IMEs) and works reliably
    #     """
    #     try:
    #         if not text:
    #             return True
                
    #         # Save current clipboard content
    #         original_clipboard = None
    #         try:
    #             win32clipboard.OpenClipboard()
    #             if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
    #                 original_clipboard = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
    #             win32clipboard.CloseClipboard()
    #         except:
    #             pass
            
    #         # Set text to clipboard
    #         win32clipboard.OpenClipboard()
    #         win32clipboard.EmptyClipboard()
    #         win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
    #         win32clipboard.CloseClipboard()
            
    #         # Paste using Ctrl+V
    #         await asyncio.sleep(0.05)  # Small delay
    #         pyautogui.hotkey('ctrl', 'v')
    #         await asyncio.sleep(0.1)  # Wait for paste to complete
            
    #         # Restore original clipboard content
    #         if original_clipboard is not None:
    #             try:
    #                 win32clipboard.OpenClipboard()
    #                 win32clipboard.EmptyClipboard()
    #                 win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, original_clipboard)
    #                 win32clipboard.CloseClipboard()
    #             except:
    #                 pass
            
    #         return True
    #     except Exception as e:
    #         logger.error(f"Error typing text '{text}': {e}")
    #         return False
    # async def type_text(self, text: str) -> bool:
    #     """使用SendMessage直接发送文本到窗口"""
    #     hwnd = win32gui.GetForegroundWindow()
    #     try:
    #         if not hwnd or not win32gui.IsWindow(hwnd):
    #             return False
            
    #         edit_hwnd = win32gui.FindWindowEx(hwnd, 0, "Edit", None)
    #         if edit_hwnd:
    #             win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, "")
    #             win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, text)
    #             return True
            
    #         for char in text:
    #             win32gui.SendMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
    #             await asyncio.sleep(0.01)  
            
    #         return True
            
    #     except Exception as e:
    #         logger.error(f"SendMessage输入失败: {e}")
    #         return False
    async def type_text(self, text: str) -> bool:
        """使用SendMessage直接发送文本到窗口"""
        hwnd = win32gui.GetForegroundWindow()
        try:
            if not hwnd or not win32gui.IsWindow(hwnd):
                logger.warning(f"无效的窗口句柄: {hwnd}")
                return False
            
            logger.info(f"目标窗口: {win32gui.GetWindowText(hwnd)} (句柄: {hwnd})")
            
            # ========== 方案1：尝试找到Edit控件直接设置文本 ==========
            edit_hwnd = win32gui.FindWindowEx(hwnd, 0, "Edit", None)
            if edit_hwnd:
                logger.info(f"找到Edit控件: {edit_hwnd}")
                
                # 先激活窗口
                win32gui.SetForegroundWindow(hwnd)
                await asyncio.sleep(0.1)
                
                # 方法1：直接设置文本
                try:
                    # 先清空
                    win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, "")
                    await asyncio.sleep(0.01)
                    # 设置新文本
                    result = win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, text)
                    logger.info(f"设置Edit文本成功，返回值: {result}")
                    return True
                except Exception as e:
                    logger.warning(f"直接设置Edit文本失败: {e}")
                    # 继续尝试其他方法
            
            # ========== 方案2：查找所有可能的编辑控件 ==========
            # 不同的应用程序可能使用不同的控件类名
            edit_classes = ["Edit", "RichEdit", "RichEdit20W", "RICHEDIT50W", 
                        "TextBox", "WindowsForms10.EDIT", "TEdit"]
            
            for edit_class in edit_classes:
                try:
                    edit_hwnd = win32gui.FindWindowEx(hwnd, 0, edit_class, None)
                    if edit_hwnd:
                        logger.info(f"找到{edit_class}控件: {edit_hwnd}")
                        
                        # 先激活窗口
                        win32gui.SetForegroundWindow(hwnd)
                        await asyncio.sleep(0.1)
                        
                        # 发送WM_SETTEXT
                        win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, "")
                        win32gui.SendMessage(edit_hwnd, win32con.WM_SETTEXT, 0, text)
                        logger.info(f"通过{edit_class}设置文本成功")
                        return True
                except:
                    continue
            
            # ========== 方案3：通过Tab键切换到输入控件 ==========
            logger.info("未找到标准编辑控件，尝试模拟按键输入")
            
            # 确保窗口激活
            win32gui.SetForegroundWindow(hwnd)
            await asyncio.sleep(0.2)
            
            # 先按Tab键尝试切换到输入框
            pyautogui.press('tab')
            await asyncio.sleep(0.05)
            
            # 尝试发送字符
            for char in text:
                try:
                    # 发送到窗口
                    win32gui.SendMessage(hwnd, win32con.WM_CHAR, ord(char), 0)
                except:
                    # 备用方案：使用keybd_event
                    if char == '\n':  # 回车
                        pyautogui.press('enter')
                    elif char == '\t':  # 制表符
                        pyautogui.press('tab')
                    else:
                        pyautogui.write(char)
                
                await asyncio.sleep(0.01)  # 小延迟避免太快
            
            logger.info(f"通过WM_CHAR发送文本完成: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"SendMessage输入失败: {e}")
            return False
    
    async def press_key(self, key: str) -> bool:
        """Press a single key"""
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            logger.error(f"Error pressing key '{key}': {e}")
            return False

    async def press_hotkey(self, key1: str, key2: str, key3: Optional[str] = None) -> bool:
        """Press a combination of keys"""
        try:
            if key3 is not None:
                pyautogui.keyDown(key1)
                pyautogui.keyDown(key2)
                pyautogui.keyDown(key3)
                pyautogui.keyUp(key3)
                pyautogui.keyUp(key2)
                pyautogui.keyUp(key1)
            else:
                pyautogui.keyDown(key1)
                pyautogui.keyDown(key2)
                pyautogui.keyUp(key2)
                pyautogui.keyUp(key1)    
            return True
        except Exception as e:
            logger.error(f"Error pressing hotkey {key1, key2, key3}: {e}")
            return False

    async def take_screenshot(self, save_path: Optional[str] = None) -> bool:
        """Take a screenshot"""
        try:
            screenshot = pyautogui.screenshot()
            if save_path:
                screenshot.save(save_path)
            return True
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return False
    
    async def get_mouse_position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        try:
            return pyautogui.position()
        except Exception as e:
            logger.error(f"Error getting mouse position: {e}")
            return (0, 0)

    async def move_mouse(self, x: float, y: float, duration: float = 0.0) -> bool:
        """Move mouse to specified position"""
        if (0 <= x <= 1 and 0 <= y <= 1):
            # If the coordinates are normalized (0 to 1)
            pos_x = self.getscreen_size()[0] * x
            pos_y = self.getscreen_size()[1] * y
        else:
            # If the coordinates are normalized (0 to 1000)
            pos_x = self.getscreen_size()[0] * (x / 1000)
            pos_y = self.getscreen_size()[1] * (y / 1000)
        try:
            pyautogui.moveTo(pos_x, pos_y, duration=duration)
            return True
        except Exception as e:
            logger.error(f"Error moving mouse to ({x}, {y}): {e}")
            return False