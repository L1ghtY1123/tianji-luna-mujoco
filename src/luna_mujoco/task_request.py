"""Literal request candidates; JEV decides their task meaning."""

import re

COLORS = {'black': '黑', 'red': '红', 'blue': '蓝', 'green': '绿',
          'yellow': '黄', 'white': '白', 'orange': '橙', 'gray': '灰'}
DIGITS = dict(zip('零〇一二两三四五六七八九', (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
UNITS = {'十': 10, '百': 100, '千': 1000}
ENGLISH_NUMBERS = dict(zip(('one', 'two', 'three', 'four', 'five', 'six', 'seven',
                          'eight', 'nine', 'ten'), range(1, 11)))


def quantity_candidates(instruction):
    numbers = []
    for match in re.finditer(r'\d+|[零〇一二两三四五六七八九十百千]+', instruction):
        token = match.group()
        if token.isascii():
            value = int(token)
        else:
            value, digit = 0, 0
            for character in token:
                if character in UNITS:
                    value += (digit or 1) * UNITS[character]
                    digit = 0
                else:
                    digit = DIGITS[character]
            value += digit
        if value > 0:
            numbers.append(value)
    for word in re.findall(r'\b[a-z]+\b', instruction.lower()):
        if word in ENGLISH_NUMBERS:
            numbers.append(ENGLISH_NUMBERS[word])
    for _ in re.finditer(r'单个|\b(?:an?\s+single|a|an|single)\b', instruction.lower()):
        numbers.append(1)
    # Offer combinations as candidates only: JEV determines which literals are
    # target counts rather than environmental counts. Keep the choice bounded.
    sums = {0}
    for value in numbers:
        combined = sums | {total + value for total in sums}
        if len(combined) > 64:
            break
        sums = combined
    return sorted({*numbers, *([sum(numbers)] if numbers else []), *(sums - {0})})


def quantity_group_candidates(instruction):
    """Literal clauses only; JEV decides whether each count constrains targets."""
    groups = []
    for clause in re.split(r'[，,;；。]|以及|和|与|\band\b', instruction, flags=re.IGNORECASE):
        clause = clause.strip()
        counts = quantity_candidates(clause)
        if counts:
            groups.append({'group_id': 'group_' + str(len(groups)), 'source_text': clause,
                           'count_candidates': counts})
    return groups if len(groups) > 1 else []


def request_scope_questions(instruction):
    counts = quantity_candidates(instruction)
    questions = {
        'quantity': {'instructions': '只提取原文明确写出的搬运数量，不预测现场有几个目标。'
                     '颜色、形状或“物体”这个称呼本身不表示数量为1。未写数量或要求全部时选unbounded，'
                     '后续根据观测和原文确认任务成员；unbounded不表示可以搬运无关物体。'
                     '数字必须修饰待搬物体，不能取环境障碍物、容器、尺寸、步骤或编号的数字。'
                     '单个、one、a/an、single修饰一个待搬物体时计1；分别描述的多个单数目标需要相加，'
                     'a single是一个单数表达，不计两次。多个明确数量组可以求和；同一个物体经桌面中转只计一次。',
                     'criteria': {'unbounded': '原文没有明确的总数量要求，或要求全部/若干；仅有颜色、形状、来源或目的地不构成数量要求',
                                  **{str(n): (f'数量词明确修饰待搬物体，合计{n}个不同目标；不累加盒子、障碍物或搬运次数。'
                                              + ('“一个物体”“同一个物体”明确计1个，即使搬运多次；仅说某颜色物体不等于指定1个。'
                                                 if n == 1 else '只有明确要求搬运的各组物体数量才可以求和。'))
                                      for n in counts}}},
        'selection_order': {'instructions': '只判断不同待搬物体之间是否有明确的执行先后或优先级。'
                            '颜色、形状、编号、相对位置、数量和来源限定哪些物体属于任务，不自动指定顺序；'
                            '并列列举的文字顺序不是执行顺序。同一物体先放桌面再放盒子是路线阶段，不是物体间顺序。'
                            '没有物体间顺序时，按确认成员清单的display_id从小到大执行。',
                            'criteria': {'inventory': '未要求不同物体的执行先后；属性限定集合或个体后，按既定清单顺序。',
                                         'explicit': '明确要求物体或分组之间的先后、优先级或遍历方向；按原文顺序执行。'}}}
    # No literal count means no count restriction to interpret. SAM3 supplies
    # observed instances; JEV still determines membership from the full request.
    if not counts:
        questions.pop('quantity')
    for index, group in enumerate(quantity_group_candidates(instruction)):
        questions[f'quantity_group_{index}'] = {
            'instructions': '独立根据完整原文判断这个原文片段是否限定一组待搬物体数量：'
                + group['source_text'] + '。环境、盒子、尺寸、步骤和同物体重复搬运的数字选not_target；'
                '不要依赖本次其他答案。多个组分别保留，后续逐组验证，不能用其他组物体补数。',
            'criteria': {'not_target': '不是独立待搬对象组的数量约束',
                         **{str(count): f'此片段明确要求这一组搬运{count}个不同物体'
                            for count in group['count_candidates']}}}
    for color, chinese in sorted(COLORS.items()):
        if chinese not in instruction and not re.search(r'\b' + color + r'\b', instruction.lower()):
            continue
        questions['environment_' + color] = {
            'instructions': f'根据完整原指令，是否提及{chinese}色普通物体作为现场环境，而非本次搬运目标？'
                            '此问题只决定额外的普通物体感知查询。区分颜色修饰的实体及其角色：'
                            '陈述现场有某些物体而未要求搬运它们，也属于环境描述，不要求出现“障碍”或“不动”。'
                            '搬运目标、来源盒、目的盒和桌面标记不属于此查询；盒子由容器查询处理。'
                            '不能因有此颜色的盒子就推测还有同色普通物体。',
            'criteria': {'yes': f'原文描述现场有{chinese}色普通物体但未要求搬它，或明确称其为障碍；该实体不是盒子或标记',
                         'no': f'未声明{chinese}色普通环境物体；该颜色未出现或只修饰搬运目标、来源/目的盒、桌面标记'}}
    return questions


def normalize_request_scope(result, *, instruction=''):
    quantity = result.setdefault('quantity', 'unbounded')
    result['requested_count'] = None if quantity == 'unbounded' else int(quantity)
    result['environment_queries'] = [
        {'kind': 'object', 'color': color, 'prompt': color + ' object'}
        for color in sorted(COLORS) if result.get('environment_' + color) == 'yes']
    result['quantity_groups'] = []
    for index, group in enumerate(quantity_group_candidates(instruction)):
        count = result.get(f'quantity_group_{index}', 'not_target')
        if count != 'not_target':
            result['quantity_groups'].append({'group_id': group['group_id'],
                'source_text': group['source_text'], 'requested_count': int(count)})
    return result
