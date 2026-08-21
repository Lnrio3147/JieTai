#!/usr/bin/env python3
"""Generate the unified RT-IGEV/LiteAnyStereo reevaluation PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/reports/JMP_LITEANYSTEREO_TEST_REPORT_zh-CN.pdf"
FONT_PATH = Path("/usr/share/fonts/truetype/arphic/ukai.ttc")
RESULT = ROOT / "runs/evaluation/jmp_unified_rerun_73"
BLUE = colors.HexColor("#2878B5")
DARK = colors.HexColor("#1F2933")
GRAY = colors.HexColor("#667085")
LIGHT_BLUE = colors.HexColor("#EAF3F9")
LIGHT_GRAY = colors.HexColor("#F3F5F6")


def report_image(path, width, max_height=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    image = Image(str(path))
    height = width * image.imageHeight / image.imageWidth
    if max_height is not None and height > max_height:
        height = max_height
        width = height * image.imageWidth / image.imageHeight
    image.drawWidth = width
    image.drawHeight = height
    image.hAlign = "CENTER"
    return image


def page_number(canvas, document):
    canvas.saveState()
    canvas.setFont("CN", 8)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(A4[0] / 2, 10 * mm, f"LiteAnyStereo 与 RT-IGEV 统一基准复评  |  {document.page}")
    canvas.restoreState()


def three_line_table(data, widths, *, highlight_row=None, font_size=9.0):
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "CN"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEABOVE", (0, 0), (-1, 0), 1.2, DARK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.65, DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 1.2, DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    if highlight_row is not None:
        commands.append(("TEXTCOLOR", (0, highlight_row), (-1, highlight_row), BLUE))
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle(commands))
    return table


def two_comparisons(scene_a, text_a, scene_b, text_b, caption):
    def image(scene):
        return report_image(RESULT / "comparisons" / scene / "traditional_comparison.png", 74 * mm, 88 * mm)

    images = Table(
        [[image(scene_a), image(scene_b)]],
        colWidths=[78 * mm, 78 * mm],
        style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]),
    )
    labels = Table(
        [[Paragraph(text_a, caption), Paragraph(text_b, caption)]],
        colWidths=[78 * mm, 78 * mm],
    )
    return [images, labels]


def build_report():
    pdfmetrics.registerFont(TTFont("CN", str(FONT_PATH)))
    title = ParagraphStyle("Title", fontName="CN", fontSize=24, leading=36, textColor=DARK, alignment=TA_CENTER)
    subtitle = ParagraphStyle("Subtitle", fontName="CN", fontSize=12, leading=20, textColor=GRAY, alignment=TA_CENTER)
    heading = ParagraphStyle("Heading", fontName="CN", fontSize=16, leading=24, textColor=BLUE, spaceBefore=4 * mm, spaceAfter=3 * mm)
    subheading = ParagraphStyle("Subheading", fontName="CN", fontSize=12, leading=18, textColor=DARK, spaceBefore=3 * mm, spaceAfter=2 * mm)
    body = ParagraphStyle("Body", fontName="CN", fontSize=10.2, leading=18, textColor=DARK, spaceAfter=2.5 * mm)
    bullet = ParagraphStyle("Bullet", parent=body, leftIndent=6 * mm, firstLineIndent=-4 * mm)
    note = ParagraphStyle("Note", parent=body, fontSize=9.2, leading=15, textColor=GRAY, backColor=LIGHT_GRAY, borderPadding=7)
    caption = ParagraphStyle("Caption", parent=body, fontSize=9.0, leading=14, alignment=TA_CENTER, textColor=GRAY)
    final = ParagraphStyle("Final", parent=body, fontSize=14, leading=22, alignment=TA_CENTER, textColor=BLUE)

    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="LiteAnyStereo 与一期 RT-IGEV 统一基准复评报告",
        author="LiteAnyStereo unified evaluation",
    )
    width = A4[0] - 36 * mm

    all73 = [
        ["算法", "EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"],
        ["RT-IGEV", "4.6745", "10.64", "40.65", "19.78", "12.65"],
        ["LiteAnyStereo", "2.0762", "7.47", "40.11", "17.44", "9.89"],
    ]
    fixed69 = [
        ["算法", "EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"],
        ["RT-IGEV", "3.4483", "8.90", "38.68", "17.35", "10.29"],
        ["LiteAnyStereo", "1.9457", "7.03", "38.86", "16.02", "8.77"],
    ]
    fixed_change = [
        ["变化", "EPE", "D1", "Bad1", "Bad2", "Bad3"],
        ["绝对变化", "-1.5026 px", "-1.87 pp", "+0.18 pp", "-1.33 pp", "-1.52 pp"],
        ["相对变化", "改善 43.58%", "改善 21.05%", "变差 0.47%", "改善 7.67%", "改善 14.79%"],
    ]
    reflection15 = [
        ["算法", "EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"],
        ["RT-IGEV", "4.1342", "13.83", "45.16", "22.93", "16.73"],
        ["LiteAnyStereo", "2.6720", "10.95", "43.60", "20.01", "13.71"],
    ]
    reflection13 = [
        ["算法", "EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"],
        ["RT-IGEV", "2.5439", "10.70", "41.30", "17.53", "11.50"],
        ["LiteAnyStereo", "2.5159", "10.14", "41.28", "16.45", "10.79"],
    ]
    reflection_pixels13 = [
        ["算法", "EPE (px)", "D1 (%)", "Bad1 (%)", "Bad2 (%)", "Bad3 (%)"],
        ["RT-IGEV", "0.8804", "0.35", "33.90", "8.66", "1.32"],
        ["LiteAnyStereo", "0.8044", "0.47", "30.76", "5.62", "0.83"],
    ]
    runtime = [
        ["方法", "设备/输入", "设置", "平均耗时", "FPS"],
        ["RT-IGEV（一期）", "A6000 / 1280×720", "文档记录 12 次迭代", "未保存", "无法计算"],
        ["LAS1 FP32", "RTX 4090 / 1280×720", "填充至 1280×736", "28.73 ms", "34.81"],
        ["LAS1 FP16 AMP", "RTX 4090 / 1280×720", "填充至 1280×736", "22.01 ms", "45.43"],
    ]

    story = [
        Spacer(1, 38 * mm),
        Paragraph("LiteAnyStereo 与一期 RT-IGEV<br/>统一基准复评报告", title),
        Spacer(1, 8 * mm),
        Paragraph("同一参考 · 同一 ROI · 同一掩码 · 同一指标实现 · 同一场景集合", subtitle),
        Spacer(1, 23 * mm),
        Table(
            [
                ["当前模型", "LiteAnyStereo LAS1 官方权重，本次重新推理 73 场"],
                ["一期结果", "RT-IGEV 保存的原始浮点 disp.npy"],
                ["全量统计", "73 场，不排除、不做 EPE 过滤"],
                ["固定统计", "排除旧工程指定 4 场后的 69 场"],
                ["报告日期", "2026-08-13"],
            ],
            colWidths=[34 * mm, 110 * mm],
            style=TableStyle([
                ("FONTNAME", (0, 0), (-1, -1), "CN"), ("FONTSIZE", (0, 0), (-1, -1), 10.2),
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE), ("TEXTCOLOR", (0, 0), (0, -1), BLUE),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD2D9")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Spacer(1, 22 * mm),
        Paragraph("结论：统一复评后，LiteAnyStereo 整体效果优于一期 RT-IGEV。", ParagraphStyle("CoverConclusion", parent=body, fontSize=13, leading=22, alignment=TA_CENTER, textColor=BLUE)),
        PageBreak(),
        Paragraph("1. 复评原因与方法", heading),
        Paragraph("旧 IGEV_metrics.csv 缺少 0018 场景，并且部分数值与 igev_output/disp.npy 不对应；同时旧 CSV 和 LiteAnyStereo 使用了不同的过滤集合。因此上一版直接平均旧 CSV 的结果不具有可比性。", note),
        Paragraph("本次不再使用旧 CSV 汇总值。RT-IGEV 直接读取 igev_output/<scene>/disp.npy 并固定裁剪 [234:1052,126:638]；LiteAnyStereo 使用官方权重重新推理并保存浮点 disp.npy。两者再进入同一个评价函数。", body),
        three_line_table([
            ["评价项", "统一设置"],
            ["参考", "Foundation Stereo disp_cropped.npy"],
            ["尺寸", "818×512"],
            ["有效掩码", "参考视差有限且大于 0"],
            ["指标", "EPE、D1、Bad1、Bad2、Bad3；同一函数"],
            ["汇总", "逐场景计算后做场景宏平均"],
            ["EPE 场景过滤", "关闭"],
        ], [42 * mm, 113 * mm], font_size=9.0),
        Spacer(1, 5 * mm),
        Paragraph("本机没有找到一期 RT-IGEV 代码和 checkpoint，因此没有重新执行旧网络前向；但其 73 场原始浮点预测完整，可以严格统一指标计算。LiteAnyStereo 已实际重新推理。", body),
        PageBreak(),
        Paragraph("2. 全部 73 场结果", heading),
        Paragraph("表 1  全 73 场宏平均（三线表）", subheading),
        three_line_table(all73, [42 * mm, 23 * mm, 21 * mm, 22 * mm, 22 * mm, 22 * mm], highlight_row=2),
        Spacer(1, 7 * mm),
        report_image(RESULT / "unified_comparison_all73.png", width),
        Paragraph("图 1  全 73 场统一指标对比", caption),
        Paragraph("LiteAnyStereo 的 EPE、D1、Bad1、Bad2、Bad3 分别改善 55.59%、29.81%、1.34%、11.85% 和 21.87%。全量评价中五项指标全部更好。", note),
        PageBreak(),
        Paragraph("3. 固定 69 场结果", heading),
        Paragraph("表 2  固定 69 场宏平均（三线表）", subheading),
        three_line_table(fixed69, [42 * mm, 23 * mm, 21 * mm, 22 * mm, 22 * mm, 22 * mm], highlight_row=2),
        Spacer(1, 5 * mm),
        Paragraph("表 3  LiteAnyStereo 相对 RT-IGEV 的变化（三线表）", subheading),
        three_line_table(fixed_change, [42 * mm, 23 * mm, 21 * mm, 22 * mm, 22 * mm, 22 * mm], font_size=8.6),
        Spacer(1, 6 * mm),
        report_image(RESULT / "unified_comparison_fixed69.png", width),
        Paragraph("图 2  固定 69 场统一指标对比", caption),
        Paragraph("LiteAnyStereo 的 EPE、D1、Bad2、Bad3 更好；Bad1 高 0.18 个百分点，差异仅 0.47%，可视为基本持平。", note),
        PageBreak(),
        Paragraph("4. 场景分布与解释", heading),
        three_line_table([
            ["统计项", "RT-IGEV", "LiteAnyStereo"],
            ["固定 69 场平均 EPE", "3.4483", "1.9457"],
            ["固定 69 场 EPE 中位数", "1.9370", "1.5728"],
            ["单场 EPE 更低数", "44", "25"],
        ], [72 * mm, 42 * mm, 42 * mm], highlight_row=2),
        Spacer(1, 6 * mm),
        Paragraph("RT-IGEV 虽在 44/69 场单场 EPE 更低，但多数领先幅度较小；LiteAnyStereo 在困难场景的改善幅度更大，因此平均值和中位数均更低。", body),
        three_line_table([
            ["场景", "RT-IGEV EPE", "LiteAnyStereo EPE", "说明"],
            ["0018", "108.8157", "12.3173", "LAS 显著改善"],
            ["0019", "55.5348", "4.6715", "全量评价；LAS 改善"],
            ["0012", "18.8266", "5.2625", "全量评价；LAS 改善"],
            ["0001", "2.0975", "0.7680", "LAS 更优"],
            ["0004", "2.3936", "0.9329", "LAS 更优"],
            ["0038", "7.4392", "8.8100", "RT-IGEV 更优反例"],
            ["0040", "2.9463", "3.5838", "RT-IGEV 更优反例"],
        ], [30 * mm, 38 * mm, 44 * mm, 45 * mm], font_size=8.5),
        Spacer(1, 5 * mm),
        Paragraph("准确结论是：LiteAnyStereo 的总体误差与困难场景鲁棒性更好，但并非每一个场景都优于 RT-IGEV。", note),
        PageBreak(),
        Paragraph("5. 金属表面高反光专项：筛选方法", heading),
        Paragraph("数据没有人工高反光标签。为避免根据模型结果挑样本，本报告仅按左图 ROI 的亮度特征，从 73 场选出高反光分数最高的 15 场，再人工确认均有明显金属亮斑。", body),
        three_line_table([
            ["亮斑特征", "定义", "分数权重"],
            ["通道截断", "max(R,G,B) ≥ 250", "1.00"],
            ["极亮", "mean(R,G,B) ≥ 220", "0.50"],
            ["近中性亮斑", "mean ≥ 200 且 max-min ≤ 35", "0.25"],
        ], [42 * mm, 77 * mm, 36 * mm], font_size=8.8),
        Spacer(1, 5 * mm),
        Paragraph("高反光分数只读取图像，不读取模型预测或误差。这是可复现的代理定义，不等同于人工材质分割。", note),
        Spacer(1, 5 * mm),
        report_image(RESULT / "high_reflection_scene_contact_sheet.png", 150 * mm, 105 * mm),
        Paragraph("图 3  高反光分数最高的 15 个金属场景", caption),
        PageBreak(),
        Paragraph("6. 金属表面高反光专项：指标", heading),
        Paragraph("表 4  高反光 15 场全 ROI 指标（三线表）", subheading),
        three_line_table(reflection15, [42 * mm, 23 * mm, 21 * mm, 22 * mm, 22 * mm, 22 * mm], highlight_row=2),
        Paragraph("LiteAnyStereo 五项指标全部更低，EPE 降低 35.37%。这 15 场包含旧协议排除的 0053 和 0020，因此还需要查看固定协议的 13 场。", body),
        Paragraph("表 5  固定协议高反光 13 场全 ROI 指标（三线表）", subheading),
        three_line_table(reflection13, [42 * mm, 23 * mm, 21 * mm, 22 * mm, 22 * mm, 22 * mm], highlight_row=2),
        Spacer(1, 5 * mm),
        report_image(RESULT / "high_reflection_fixed13_scene_comparison.png", width),
        Paragraph("图 4  固定协议高反光 13 场全 ROI 对比", caption),
        Paragraph("排除 0053 和 0020 后，LiteAnyStereo 五项仍略优：EPE 改善 1.10%，D1 改善 5.28%，Bad2/Bad3 改善 6.19%/6.23%，Bad1 基本持平。", note),
        PageBreak(),
        Paragraph("7. 高光像素专项指标", heading),
        Paragraph("固定 13 场的高光掩码平均覆盖 ROI 的 9.67%，共有 526,605 个有效高光像素。下表只在亮斑像素计算指标，再做场景宏平均。", body),
        Paragraph("表 6  固定 13 场仅高光像素指标（三线表）", subheading),
        three_line_table(reflection_pixels13, [42 * mm, 23 * mm, 21 * mm, 22 * mm, 22 * mm, 22 * mm], highlight_row=2),
        Spacer(1, 6 * mm),
        report_image(RESULT / "high_reflection_fixed13_pixel_comparison.png", width),
        Paragraph("图 5  固定协议高反光 13 场仅亮斑像素对比", caption),
        Paragraph("LiteAnyStereo 的高光像素 EPE 改善 8.64%，Bad1/2/3 改善 9.26%/35.09%/37.23%；D1 增加 0.12 个百分点。D1 还包含相对误差条件，因此与 Bad3 不完全一致。", note),
        PageBreak(),
        Paragraph("8. 高反光代表场景", heading),
    ]

    story.extend(two_comparisons(
        "202506281614-0035", "图 6  高反光场景 0035",
        "202506281614-0036", "图 7  高反光场景 0036", caption,
    ))
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("0035 中两者全 ROI 接近；0036 中 LiteAnyStereo 的 EPE 更低。六宫格已使用相同视差和误差色标。", body),
        PageBreak(),
        Paragraph("9. 更多高反光场景", heading),
    ])
    story.extend(two_comparisons(
        "202506281614-0034", "图 8  高反光场景 0034",
        "202506281605-0009", "图 9  高反光场景 0009", caption,
    ))
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("高反光结论：LiteAnyStereo 的亮斑平均误差和 Bad1/2/3 总体更低，但 D1 略高，且排除异常场景后全 ROI 优势较小，不能表述为已经完全解决金属反光问题。", note),
        PageBreak(),
        Paragraph("10. 统一色标视差与误差图", heading),
        Paragraph("六宫格依次为左图 ROI、RT-IGEV 视差、LiteAnyStereo 视差、参考视差、RT-IGEV 误差和 LiteAnyStereo 误差。视差统一为 0–192 px，误差统一为 0–20 px。", body),
    ])

    story.extend(two_comparisons(
        "202506281603-0001", "图 10  0001：2.0975 → 0.7680 px",
        "202506281604-0004", "图 11  0004：2.3936 → 0.9329 px", caption,
    ))
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("这两个常规场景中 LiteAnyStereo 的主体区域和边界误差均更低。", body),
        PageBreak(),
        Paragraph("11. 困难场景改善与反例", heading),
    ])
    story.extend(two_comparisons(
        "202506281608-0018", "图 12  0018：108.8157 → 12.3173 px",
        "202506281615-0038", "图 13  0038：7.4392 → 8.8100 px", caption,
    ))
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("0018 中 LiteAnyStereo 避免了 RT-IGEV 的灾难性失效；0038 是 RT-IGEV 更优的反例。", body),
        PageBreak(),
        Paragraph("12. 更多反例", heading),
    ])
    story.extend(two_comparisons(
        "202506281616-0040", "图 14  0040：2.9463 → 3.5838 px",
        "202506281613-0030", "图 15  0030：2.4885 → 2.6387 px", caption,
    ))
    story.extend([
        Spacer(1, 5 * mm),
        Paragraph("这两个场景中 RT-IGEV 更好，说明 LiteAnyStereo 仍需针对部分常规纹理和边界区域优化。", body),
        PageBreak(),
        Paragraph("13. 推理时间与结论", heading),
        Paragraph("表 7  当前能够核实的核心推理时间（三线表）", subheading),
        three_line_table(runtime, [34 * mm, 46 * mm, 40 * mm, 24 * mm, 18 * mm], font_size=8.5),
        Spacer(1, 6 * mm),
        Paragraph("一期 RT-IGEV 没有留下项目推理时间日志，因此不能给出严格速度倍数。LiteAnyStereo 本机核心前向为 28.73 ms/对（FP32）或 22.01 ms/对（FP16 AMP）。", note),
        Spacer(1, 7 * mm),
        Paragraph("• 统一评价后，LiteAnyStereo 在全 73 场五项指标全部更好。", bullet),
        Paragraph("• 固定 69 场中 EPE 改善 43.58%，D1 改善 21.05%，Bad2 改善 7.67%，Bad3 改善 14.79%。", bullet),
        Paragraph("• 固定 69 场 Bad1 高 0.18 个百分点，与 RT-IGEV 基本持平。", bullet),
        Paragraph("• LiteAnyStereo 对困难和异常场景更稳，但 RT-IGEV 仍在部分单场占优。", bullet),
        Paragraph("• 高反光固定 13 场中两者全 ROI 接近，LiteAnyStereo 五项略优；高光像素的 EPE 和 Bad1/2/3 更好，但 D1 高 0.12 个百分点。", bullet),
        Paragraph("• 上一版异常结论来自旧 CSV、原始输出和过滤集合不一致，本次浮点视差统一复评应作为最终口径。", bullet),
        Spacer(1, 8 * mm),
        Paragraph("最终判断：LiteAnyStereo 整体效果优于一期 RT-IGEV，但仍需继续优化 Bad1 和部分 RT-IGEV 占优场景。", final),
    ])

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
