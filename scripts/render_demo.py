"""Render evidence summaries only. No Agent, external requests or dependencies."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def render(data):
    if data.get('mode') != 'static_evidence_summary' or data.get('not_live_agent') is not True:
        raise ValueError('Only explicitly labelled static evidence may be rendered')
    escape = html.escape
    cards = []
    for item in data['scenarios']:
        steps = ''.join('<li>'+escape(step)+'</li>' for step in item['steps'])
        cards.append('<article><p class="tag">'+escape(item['evidence_kind'])+'</p><h2>'+escape(item['title'])+
                     '</h2><p>'+escape(item['input'])+'</p><ol>'+steps+'</ol><p class="limit">边界：'+
                     escape(item['limitation'])+'</p><details><summary>证据定位</summary><p>'+escape(item['locator'])+
                     '</p><p>索引：'+escape(', '.join(item['source_ids']))+'</p></details></article>')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>客服 Agent · 工程证据演示</title><style>
body{font:16px/1.75 system-ui,sans-serif;background:#f4f6fa;color:#1d2939;margin:0}main{max-width:1060px;margin:auto;padding:38px 24px}
h1{font-size:30px;line-height:1.3}h2{font-size:20px}header{border-left:5px solid #176b63;padding-left:22px;margin-bottom:24px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}article{background:white;border:1px solid #dce3e9;border-radius:12px;padding:22px}
.tag{font:12px monospace;color:#176b63}.limit{background:#fff5df;padding:10px;border-radius:6px;font-size:14px}details{font-size:13px;color:#475467}
footer{margin-top:24px;padding:18px;background:#e9eef4;border-radius:8px}li{margin-bottom:7px}@media print{body{background:white}article{break-inside:avoid}.grid{display:block}article{margin-bottom:12px}}
</style><main><header><p>电商智能客服 Agent · 基于 LangGraph 的可靠性与安全增强</p>
<h1>四个场景，讲清工程机制和证据边界</h1>
<p>静态证据摘要 · 不是实时 Agent · 不调用模型 · 不含原始 Trace 或课程源码</p></header><section class="grid">'''+''.join(cards)+'''</section>
<footer><strong>保留失败：</strong>历史 12 个合成任务 / 13 轮业务评分通过，但当时预算保护审计不合规。
后续缺测停止与流式状态隔离仅完成离线验证，不能当作当前在线通过。项目基于购买的 MewHelp 课程二次开发，使用 Codex 辅助。</footer></main></html>'''

def main():
    data = json.loads((ROOT/'demo/scenarios.json').read_text(encoding='utf8'))
    (ROOT/'demo/index.html').write_text(render(data), encoding='utf8')
    print('Rendered demo/index.html: static evidence only; external requests=0')

if __name__ == '__main__':
    main()
