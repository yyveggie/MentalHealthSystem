#!/usr/bin/env python3
"""
启动脚本 - 同时运行HTTP API和WebSocket服务
"""
import asyncio
import threading
import time
from flask import Flask, request, json
from datetime import datetime

# 简化的Flask应用
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title> 医院 AI 心理治疗系统</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .status {{ background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .api-info {{ background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            .feature {{ margin: 10px 0; padding: 10px; background: #fafafa; border-left: 4px solid #3498db; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥  医院 AI 心理治疗系统</h1>
            
            <div class="status">
                <h3>🟢 系统状态：运行中</h3>
                <p>服务器时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>HTTP API 状态：正常</p>
                <p>WebSocket 状态：已禁用</p>
            </div>
            
            <div class="api-info">
                <h3>📡 服务信息</h3>
                <p><strong>HTTP API服务：</strong> http://127.0.0.1:8763</p>
                <p><strong>WebSocket服务：</strong> 已禁用</p>
                <p><strong>医疗诊断接口：</strong> POST /apiv1/diagnosis/processor</p>
                <p><strong>认证Token：</strong> <a href="/tokens">获取Token</a></p>
            </div>
            
            <h3>🧠 系统功能</h3>
            <div class="feature">
                <strong>心理咨询对话系统：</strong> WebSocket 实时对话
            </div>
            <div class="feature">
                <strong>医疗诊断系统：</strong> 基于病历数据的智能诊断
            </div>
            
            <div style="text-align: center; margin-top: 30px;">
                <a href="/tokens" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">获取认证Token</a>
                <a href="/api-docs" style="background: #28a745; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-left: 10px;">API文档</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/status', methods=['GET'])
def status():
    """系统状态检查"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
    except ImportError:
        cpu_percent = "N/A"
        memory_percent = "N/A" 
        disk_percent = "N/A"
    
    status_info = {
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent
        },
        "services": {
            "http_api": "running",
            "websocket": "disabled"
        },
        "version": "1.0.0"
    }
    
    return json.dumps(status_info, ensure_ascii=False, indent=2)

@app.route('/tokens', methods=['GET'])
def tokens():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>认证Token</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
            .token {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔐 认证Token</h1>
            <p>以下是可用的认证Token，请复制其中任意一个使用：</p>
            
            <div class="token">
                <strong>演示Token:</strong><br>
                <code>demo-token-123</code>
            </div>
            
            <div class="token">
                <strong>测试Token:</strong><br>
                <code>test-token-456</code>
            </div>
            
            <div class="token">
                <strong>管理员Token:</strong><br>
                <code>admin-token-789</code>
            </div>
            
            <div class="token">
                <strong>医疗API Token:</strong><br>
                <code>medical-api-2025</code>
            </div>
            
            <p><a href="/">← 返回首页</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/api-docs', methods=['GET'])
def api_docs():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API文档</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .code { background: #f4f4f4; padding: 15px; border-radius: 5px; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📖 API文档</h1>
            
            <h3>医疗诊断API</h3>
            <p><strong>接口地址:</strong> <code>POST /apiv1/diagnosis/processor</code></p>
            <p><strong>认证:</strong> 需要 X-Ivanka-Token 请求头</p>
            
            <h4>请求示例:</h4>
            <div class="code">
curl -X POST http://127.0.0.1:8763/apiv1/diagnosis/processor \\
  -H "Content-Type: application/json" \\
  -H "X-Ivanka-Token: demo-token-123" \\
  -d '{"主诉": "头痛失眠3个月", "现病史": "患者头痛伴失眠"}'
            </div>
            
            <p><a href="/">← 返回首页</a></p>
        </div>
    </body>
    </html>
    """

@app.route('/apiv1/diagnosis/processor', methods=['POST'])
def diagnosis_processor():
    try:
        fields = request.get_json(force=True)
        token = request.headers.get('X-Ivanka-Token')
        
        if not token:
            return json.dumps({"error": "TOKEN为空"}, ensure_ascii=False)
        
        valid_tokens = ["demo-token-123", "test-token-456", "admin-token-789", "medical-api-2025"]
        if token not in valid_tokens:
            return json.dumps({"error": "无效的TOKEN"}, ensure_ascii=False)
        
        # 使用真正的医疗诊断处理器
        try:
            # 导入并使用真正的诊断处理器
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            
            from business.diagnose import MedicalDiagnosisProcessor
            
            processor = MedicalDiagnosisProcessor()
            result = processor.process_diagnosis(fields)
            
            if "error" in result:
                # AI诊断失败，使用增强版规则引擎
                print("AI诊断返回错误，使用增强版规则引擎")
                return enhanced_rule_based_diagnosis(fields)
            
            # 格式化输出
            resp = processor.output_format(raw_results=result)
            return json.dumps(resp, ensure_ascii=False)
            
        except Exception as ai_error:
            print(f"AI诊断失败，使用增强版规则引擎: {str(ai_error)}")
            
            # AI诊断失败时，使用更智能的规则引擎
            return enhanced_rule_based_diagnosis(fields)
        
    except Exception as e:
        return json.dumps({"error": f"处理出错: {str(e)}"}, ensure_ascii=False)

def enhanced_rule_based_diagnosis(fields):
    """增强版规则引擎诊断"""
    import re
    from datetime import datetime
    
    # 合并所有文本进行分析
    all_text = " ".join(fields.values()).lower()
    
    # 症状关键词映射
    symptom_patterns = {
        "头痛": {
            "keywords": ["头痛", "头疼", "偏头痛", "头晕", "头胀"],
            "diagnoses": [
                {"病症": "偏头痛", "置信度": 0.75, "理由": "根据头痛症状特征，考虑偏头痛可能"},
                {"病症": "紧张性头痛", "置信度": 0.70, "理由": "持续性头痛，可能为紧张性头痛"}
            ]
        },
        "失眠": {
            "keywords": ["失眠", "睡不着", "入睡困难", "早醒", "睡眠质量差"],
            "diagnoses": [
                {"病症": "失眠症", "置信度": 0.80, "理由": "根据睡眠障碍症状，诊断为失眠症"},
                {"病症": "焦虑性失眠", "置信度": 0.65, "理由": "失眠可能与焦虑情绪相关"}
            ]
        },
        "胸闷": {
            "keywords": ["胸闷", "胸痛", "胸部不适", "呼吸困难", "气短"],
            "diagnoses": [
                {"病症": "焦虑症", "置信度": 0.78, "理由": "胸闷、气短等症状常见于焦虑症"},
                {"病症": "心脏神经官能症", "置信度": 0.72, "理由": "胸部症状可能为功能性心脏疾病"}
            ]
        },
        "心慌": {
            "keywords": ["心慌", "心悸", "心跳快", "心律不齐"],
            "diagnoses": [
                {"病症": "心律失常", "置信度": 0.75, "理由": "心慌、心悸症状提示可能的心律失常"},
                {"病症": "焦虑症", "置信度": 0.70, "理由": "心慌症状常伴随焦虑情绪"}
            ]
        },
        "抑郁": {
            "keywords": ["抑郁", "情绪低落", "兴趣减退", "无望感", "悲观"],
            "diagnoses": [
                {"病症": "抑郁症", "置信度": 0.85, "理由": "根据抑郁情绪和相关症状，考虑抑郁症诊断"},
                {"病症": "心境障碍", "置信度": 0.75, "理由": "情绪症状提示可能的心境障碍"}
            ]
        }
    }
    
    # 分析症状
    matched_diagnoses = []
    confidence_boost = 0
    
    for symptom_type, symptom_data in symptom_patterns.items():
        for keyword in symptom_data["keywords"]:
            if keyword in all_text:
                for diagnosis in symptom_data["diagnoses"]:
                    # 检查是否已存在相同诊断
                    existing = next((d for d in matched_diagnoses if d["病症"] == diagnosis["病症"]), None)
                    if existing:
                        # 提高置信度
                        existing["置信度"] = min(0.95, existing["置信度"] + 0.1)
                        existing["理由"] += f"；合并{symptom_type}症状"
                    else:
                        matched_diagnoses.append(diagnosis.copy())
                confidence_boost += 0.05
                break
    
    # 根据病史调整诊断
    if "既往史" in fields:
        past_history = fields["既往史"].lower()
        if "高血压" in past_history or "心脏病" in past_history:
            for diag in matched_diagnoses:
                if "心" in diag["病症"]:
                    diag["置信度"] = min(0.95, diag["置信度"] + 0.1)
                    diag["理由"] += "，既往心血管疾病史支持此诊断"
    
    # 根据家族史调整
    if "家族史" in fields:
        family_history = fields["家族史"].lower()
        if "精神疾病" in family_history or "抑郁" in family_history:
            for diag in matched_diagnoses:
                if "抑郁" in diag["病症"] or "焦虑" in diag["病症"]:
                    diag["置信度"] = min(0.95, diag["置信度"] + 0.1)
                    diag["理由"] += "，家族精神疾病史增加患病风险"
    
    # 如果没有匹配的诊断，提供通用建议
    if not matched_diagnoses:
        # 基于主诉进行更细致的分析
        chief_complaint = fields.get("主诉", "").lower()
        if chief_complaint:
            matched_diagnoses.append({
                "病症": "症状性疾病",
                "置信度": 0.60,
                "理由": f"根据主诉'{fields.get('主诉', '')}',需要进一步检查明确诊断"
            })
        else:
            matched_diagnoses.append({
                "病症": "需进一步检查",
                "置信度": 0.50,
                "理由": "症状描述不够详细，建议完善相关检查"
            })
    
    # 按置信度排序
    matched_diagnoses.sort(key=lambda x: x["置信度"], reverse=True)
    
    # 限制返回数量
    matched_diagnoses = matched_diagnoses[:3]
    
    result = {
        "诊断结果": matched_diagnoses,
        "session_id": f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "输入字段": list(fields.keys()),
        "处理时间": datetime.now().isoformat(),
        "状态": "成功",
        "诊断模式": "增强规则引擎"
    }
    
    return json.dumps(result, ensure_ascii=False)

def run_flask():
    """运行Flask HTTP服务器"""
    print("🚀 启动HTTP API服务器...")
    app.run(debug=False, host='0.0.0.0', port=8763, use_reloader=False)

def run_websocket():
    """运行WebSocket服务器（已禁用）"""
    print("⚠️ WebSocket服务器已禁用，跳过启动")

def main():
    print("=" * 60)
    print("🏥  医院 AI 心理治疗系统")
    print("=" * 60)
    
    # 启动HTTP服务器线程
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # 等待Flask启动
    time.sleep(2)
    
    # WebSocket已禁用，不再启动
    
    print("✅ 服务启动完成!")
    print("📡 HTTP API服务: http://127.0.0.1:8763")
    print("🔗 WebSocket服务: ws://127.0.0.1:8765")
    print("📖 API文档: http://127.0.0.1:8763/api-docs")
    print("🔐 Token管理: http://127.0.0.1:8763/tokens")
    print("-" * 60)
    print("按 Ctrl+C 退出")
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序被用户中断，正在退出...")

if __name__ == "__main__":
    main()
