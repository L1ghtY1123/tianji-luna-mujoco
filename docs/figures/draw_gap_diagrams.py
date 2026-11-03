"""Generate the two explanatory diagrams; these never generate robot plans."""
from html import escape
from pathlib import Path


class Diagram:
    def __init__(self, width, height, title):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{escape(title)}</title>',
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0 0 L9 4.5 L0 9" fill="#536b80"/></marker></defs>',
            '<style>text{font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;fill:#20364a;font-size:18px} .small{font-size:15px;fill:#526b80} .title{font-size:24px;font-weight:700} .group{font-size:20px;font-weight:700} .edge{fill:none;stroke:#536b80;stroke-width:2;marker-end:url(#arrow)}</style>',
            f'<rect width="{width}" height="{height}" fill="#fff"/>']

    def text(self, x, y, lines, style='', anchor='middle'):
        for index, line in enumerate(lines):
            self.parts.append(f'<text x="{x}" y="{y + index * 26}" text-anchor="{anchor}" class="{style}">{escape(line)}</text>')

    def box(self, x, y, width, height, lines, fill='#eaf2fb', stroke='#6587a7', dashed=False):
        dash = ' stroke-dasharray="6 4"' if dashed else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
        self.text(x + width / 2, y + (height - (len(lines) - 1) * 26) / 2 + 6, lines)

    def group(self, x, y, width, height, label, fill='#f7f9fc', stroke='#a8bacb'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        self.text(x + 20, y + 31, [label], 'group', 'start')

    def edge(self, points, label=None, at=None, dashed=False):
        path = 'M' + ' L'.join(f'{x},{y}' for x, y in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        self.parts.append(f'<path d="{path}" class="edge"{dash}/>')
        if label:
            self.text(*at, [label], 'small')

    def save(self, name):
        Path(__file__).with_name(name).write_text(''.join(self.parts) + '</svg>', encoding='utf8')


def layers():
    d = Diagram(900, 1080, '一次编排，沿图执行：控制层级')
    d.text(450, 38, ['一次编排，沿图执行'], 'title')
    d.text(450, 68, ['Graph 是可执行策略；Subgraph 不是 sub-agent'], 'small')
    d.group(30, 92, 840, 280, '01  编排层 · 只调用一次 Responses')
    d.box(90, 145, 720, 50, ['自然语言 ＋ 初始三相机 RGB ＋ 配置与技能契约'])
    d.box(340, 220, 220, 48, ['Responses'], '#eee7fb', '#8a6cad')
    d.box(90, 304, 340, 48, ['Graph：Subgraph、Node、Edge'])
    d.box(500, 304, 310, 48, ['物体清单 ＋ 各物体任务要求'], '#fff4dc', '#bfa16b')
    d.edge([(450, 195), (450, 220)])
    d.edge([(395, 268), (395, 285), (260, 285), (260, 304)])
    d.edge([(505, 268), (505, 285), (655, 285), (655, 304)])

    d.group(30, 402, 840, 360, '02  执行层 · 运行器与 Decisions')
    d.box(90, 480, 300, 88, ['运行器', '执行当前 Node，沿 Edge 推进'])
    d.box(540, 480, 270, 88, ['Decisions', '判断事实', '选择可执行恢复边'], '#eee7fb', '#8a6cad')
    d.box(540, 646, 270, 84, ['清单记忆', '最后观测、已验证关系', '该物体的操作记录'], '#fff4dc', '#bfa16b')
    d.edge([(260, 352), (260, 387), (15, 387), (15, 502), (90, 502)], '执行图', (302, 390))
    d.edge([(655, 352), (848, 352), (848, 689), (810, 689)], dashed=True)
    d.edge([(390, 502), (540, 502)], '判断请求', (465, 490))
    d.edge([(540, 552), (390, 552)], '判断结果／选边', (465, 581))
    d.edge([(675, 646), (675, 568)], '读取历史与当前证据', (675, 620), dashed=True)
    d.text(240, 638, ['顺序边：成功后继续', '条件边：根据判断分流', '回环边：补充证据后再判断'], 'small')

    d.group(30, 796, 840, 206, '03  技能层 · 每次执行被选中的能力')
    d.box(90, 860, 720, 66, ['观测　建图　抓姿　轨迹规划　抓取　放置　处理　核验', '技能内部实现固定边界，不在内部偷偷编排下一项任务'])
    d.edge([(170, 568), (170, 592), (20, 592), (20, 878), (90, 878)], '调用技能', (121, 783))
    d.edge([(90, 908), (5, 908), (5, 552), (90, 552)])
    d.text(118, 747, ['返回结果／未知／失败'], 'small', 'start')
    d.edge([(747, 860), (747, 730)], '记录证据', (790, 782), dashed=True)
    d.text(450, 966, ['数值几何、碰撞、关节限位由程序检查，不能由模型判断覆盖'], 'small')
    d.text(450, 1040, ['实线：生成、调用和控制转移　　虚线：证据读写'], 'small')
    d.save('gap-1.svg')


def objects():
    d = Diagram(900, 1450, '多物体独立编排：一个Graph包含独立Subgraph')
    d.text(450, 38, ['多物体独立编排'], 'title')
    d.text(450, 68, ['结构示意：实际节点及顺序由初始 Responses 生成'], 'small')
    d.group(25, 90, 850, 1280, 'Graph · 整项任务的总编排')
    d.box(245, 143, 410, 48, ['共享观测与清单确认节点组'])

    def item(top, name, destination, color):
        identifier = name.split('／')[0]
        d.group(55, top, 790, 460, f'Subgraph {name} · 独立物体编排', '#fbfcfe')
        d.text(78, top + 60, [f'任务要求：{destination}；不拆包。使用该物体的永久 ID。'], 'small', 'start')
        d.box(100, top + 113, 210, 55, [f'Node：绑定目标 {identifier}'], color)
        d.box(100, top + 228, 210, 85, ['其他抓放 Node', '此处折叠显示'], color, dashed=True)
        d.box(100, top + 366, 210, 55, [f'Node：核验 {identifier} 完成'], color)
        d.box(468, top + 113, 295, 55, ['Decisions 选择恢复 Edge'], '#eee7fb', '#8a6cad')
        d.box(505, top + 221, 220, 48, ['Node：建图'])
        d.box(505, top + 295, 220, 48, ['Node：换观测位姿'])
        d.box(505, top + 369, 220, 48, ['Node：重新观测'])
        d.edge([(205, top + 168), (205, top + 228)], '成功 Edge', (258, top + 203))
        d.edge([(205, top + 313), (205, top + 366)])
        d.edge([(310, top + 140), (468, top + 140)], 'unknown Edge', (389, top + 128))
        d.edge([(615, top + 168), (615, top + 221)], '选择换位姿分支', (675, top + 200))
        d.edge([(615, top + 269), (615, top + 295)])
        d.edge([(615, top + 343), (615, top + 369)])
        d.edge([(725, top + 393), (805, top + 393), (805, top + 85), (205, top + 85), (205, top + 113)])
        d.text(469, top + 101, ['回环 Edge：新证据返回目标绑定'], 'small')
        return top + 421

    last_a = item(221, '甲／粉色', '预设输入区域', '#ffe9f0')
    last_b = item(731, '乙／绿色', '桌面', '#e4f5e9')
    d.edge([(245, 167), (42, 167), (42, 361), (100, 361)])
    d.edge([(205, last_a), (205, 705), (42, 705), (42, 871), (100, 871)], '甲完成后调度乙', (302, 705))
    d.box(245, 1240, 560, 85, ['最终回位 ＋ 整体核验节点组', '各物体均满足自己的完成要求后才完成任务'])
    d.edge([(205, last_b), (205, 1282), (245, 1282)])
    d.text(450, 1410, ['共享清单，独立目标与进度；不创建 sub-agent，不按物体追加 Responses。'], 'small')
    d.save('gap-2.svg')


if __name__ == '__main__':
    layers()
    objects()
