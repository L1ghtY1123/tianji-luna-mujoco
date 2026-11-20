"""Read the explicit scene before any task observation or compilation."""
import re


def scene_sections(instruction):
    text = instruction.strip()
    first = re.match(r'(?:场景|scene)[ \t]*[:：]', text, re.IGNORECASE)
    if not first:
        raise ValueError('请先补充场景段，例如“场景：一张桌子、一个黑色 CAD 物体、两个蓝色盒子。”')
    body = text[first.end():]
    task_header = re.search(r'(?:任务|\btask)[ \t]*[:：]', body, re.IGNORECASE)
    if task_header:
        scene, task = body[:task_header.start()], body[task_header.end():]
    else:
        scene, _, task = body.strip().partition('\n')
    if not any(c.isalnum() for c in scene):
        raise ValueError('请在场景段列出允许识别的物体。')
    if not any(c.isalnum() for c in task):
        raise ValueError('请在场景段后补充任务，例如“任务：把黑色 CAD 物体放到右侧盒子。”')
    return scene.strip(), task.strip()
