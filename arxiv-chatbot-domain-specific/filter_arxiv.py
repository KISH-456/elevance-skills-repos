import json

input_path = '/home/ubuntu/upload/kaggle.json'
output_path = '/home/ubuntu/arxiv_cs.jsonl'

count = 0
cs_count = 0

with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
    for line in f_in:
        count += 1
        try:
            data = json.loads(line)
            categories = data.get('categories', '')
            if 'cs.' in categories:
                # Keep relevant fields for the chatbot
                filtered_data = {
                    'id': data.get('id'),
                    'title': data.get('title'),
                    'authors': data.get('authors'),
                    'abstract': data.get('abstract'),
                    'categories': categories,
                    'update_date': data.get('update_date')
                }
                f_out.write(json.dumps(filtered_data) + '\n')
                cs_count += 1
        except json.JSONDecodeError:
            continue

print(f"Total papers processed: {count}")
print(f"CS papers found and saved: {cs_count}")
