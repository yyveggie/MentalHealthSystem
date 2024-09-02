import json
import jieba
import re

def filter_json_by_keyword(json_data, keyword, target_key):
    """
    Filter JSON data based on keyword matching in the specified key's value.
    
    :param json_data: JSON data as a string or a list of dictionaries
    :param keyword: String to match against
    :param target_key: Key in JSON objects to check for keyword
    :return: Filtered JSON data
    """
    # Load JSON data if it's a string
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Tokenize the keyword (support for both English and Chinese)
    keyword_tokens = jieba.lcut(keyword.lower())
    
    def token_match(text):
        # Tokenize the text (support for both English and Chinese)
        text_tokens = jieba.lcut(text.lower())
        # Check if any keyword token matches any text token
        return any(k in text_tokens for k in keyword_tokens)
    
    # Filter the data
    filtered_data = [
        item for item in data
        if target_key in item and token_match(item[target_key])
    ]
    
    return json.dumps(filtered_data, ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    sample_json = '''
    [
        {"title": "Depression Treatment", "content": "Various ways to treat depression"},
        {"title": "Anxiety Disorders", "content": "Understanding anxiety and its types"},
        {"title": "Healthy Living", "content": "Tips for maintaining a healthy lifestyle"},
        {"title": "抑郁症治疗", "content": "各种治疗抑郁症的方法"},
        {"title": "焦虑障碍", "content": "理解焦虑及其类型"},
        {"title": "健康生活", "content": "保持健康生活方式的技巧"}
    ]
    '''
    
    keyword = "抑郁症 treatment"
    target_key = "title"
    
    filtered_json = filter_json_by_keyword(sample_json, keyword, target_key)
    print(filtered_json)