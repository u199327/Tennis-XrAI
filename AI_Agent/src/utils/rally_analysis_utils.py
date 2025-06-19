import csv
from src.utils.utils import init_azure_llm
from collections import Counter

def split_by_shot_type(sequence):
    shot_types = set("fbrsvzopuylmhijktq")
    segments = []
    current = ""

    for char in sequence:
        if char in shot_types:
            if current:
                segments.append(current)
            current = char 
        else:
            current += char 
    if current:
        segments.append(current)
    return segments

def load_rally_data_from_csv(file_path):
    """
    Reads rally data from a CSV file and returns a list of dictionaries,
    where each dictionary represents one rally (point).
    """
    expected_structure = [
        "match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#", 
        "TbSet", "Svr", "1st", "2nd", "Notes", "PtWinner"
    ]
    
    rally_data = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if all(field in row for field in expected_structure):
                # Strip whitespace and ensure correct formatting
                clean_row = {k: v.strip() for k, v in row.items()}
                rally_data.append(clean_row)
            else:
                print(f"Skipping row due to missing fields: {row}")
    
    return rally_data

def filter_rallies(data_lines, player_name, role="server", outcome="win"):
    """
    Filters rally data for a specific player and role (server/returner)
    and the outcome of the point (win/loss).
    """
    filtered = []
    for point in data_lines:

        server_name = point["match_id"].split('-')[-2].replace('_', ' ')
        returner_name = point["match_id"].split('-')[-1].replace('_', ' ')
        winner_id = point["PtWinner"]
        server_id = point["Svr"]
        player_id = "1" if player_name == server_name else "2"

        if role == "server" and player_name != server_name:
            continue
        if role == "returner" and player_name != returner_name:
            continue
        if outcome == "win" and player_id != winner_id:
            continue
        if outcome == "loss" and player_id == winner_id:
            continue

        first = split_by_shot_type(point["1st"].strip())
        second = split_by_shot_type(point["2nd"].strip())
        full = first + second

        filtered.append({
            "match_meta": point,
            "sequence": full,
            "server": server_name,
            "returner": returner_name
        })

    return filtered

def common_rally_sequence(data_lines, player_name, role="server", outcome="win", k=10):
    data_filtered = filter_rallies(data_lines, player_name, role, outcome)
    sequence_counter = Counter()

    for point in data_filtered:
        full_sequence = ''.join(item for item in point['sequence'] if item)
        sequence_counter[full_sequence] += 1

    # Filter out sequences that are only 2 characters long --> Mainly serves
    filtered_counter = Counter({k: v for k, v in sequence_counter.items() if len(k) > 2})

    top_10_filtered = filtered_counter.most_common(k)

    return top_10_filtered

def most_eficient_plays(data_lines, k = 10):
    sequence_counter = Counter()

    for point in data_lines:
        first = split_by_shot_type(point["1st"].strip())
        second = split_by_shot_type(point["2nd"].strip())
        full_sequence = ''.join(item for item in first + second if item)
        sequence_counter[full_sequence] += 1

    # Filter out sequences that are only 2 characters long
    filtered_counter = Counter({seq: count for seq, count in sequence_counter.items() if len(seq) > 2})

    top_10_filtered = filtered_counter.most_common(k)

    return top_10_filtered
    
def generate_rally_summary(data, player_name, role, outcome):

    strikes_counter_dict = common_rally_sequence(data, player_name, role, outcome)
    llm = init_azure_llm()

    with open('databases_tenis/instructions.txt', 'r') as file:
        instructions = file.read()

    formatted_sequence_str = "\n".join([f"{seq}: {count}" for seq, count in strikes_counter_dict])

    prompt = f"""
{instructions}

### Task
You're role is to be a Tennis Trainer.
You are given a dictionary with sequences of strokes from charted tennis points and the number of times they occur, following the coding conventions above. Your task is to translate these sequences into a natural language description.

### Input Format
The input includes:
- The player name, the role they had during the point (Server or Returner), and the outcome of the point (e.g. Won the point, Lost the point)
- A dictionary with stroke sequences and their frequency

To determine who performed each stroke:
- Elements at **odd indexes** (1, 3, 5, ...) were hit by the **server**
- Elements at **even indexes** (0, 2, 4, ...) were hit by the **returner**

### Analysis Metadata
Player: {player_name}  
Role: {role}  
Outcome: {outcome}

### Dictionary of Stroke Sequences
{formatted_sequence_str}

### Expected Output
Provide a detailed and structured using bullet points, natural language description of the most common rally stroke sequences. Focus on decoding shot direction, type, depth, and tactical patterns based on the sequence notation. After explaining the findings, give tailored advice or insights for improving performance based on the patterns observed.
Do not user charting code, with natural language is enough, since the user won't understand the coding. Limit the answer to 200 words
"""

    return llm.invoke(prompt).content
    