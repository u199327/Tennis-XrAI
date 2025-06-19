import json
from collections import defaultdict
import matplotlib.pyplot as plt
def global_accuracy(test_results):
    total_correct = sum(result['score'] for result in test_results)
    total_questions = len(test_results)

    overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

    plt.figure(figsize=(4, 4))
    plt.bar(["Overall Accuracy"], [overall_accuracy], color='skyblue')
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Overall Test Accuracy")
    plt.show()

    return overall_accuracy
def accuracy_by_difficulty(test_results):
    difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for result in test_results:
        difficulty = result['difficulty']
        score = result['score']

        difficulty_stats[difficulty]['total'] += 1
        difficulty_stats[difficulty]['correct'] += score

    correct_percentage_by_difficulty = {
        difficulty: (stats['correct'] / stats['total']) * 100
        for difficulty, stats in difficulty_stats.items()
    }

    difficulties = list(correct_percentage_by_difficulty.keys())
    percentages = list(correct_percentage_by_difficulty.values())

    plt.figure(figsize=(8, 4))
    plt.bar(difficulties, percentages, color='skyblue')
    plt.xlabel('Difficulty')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy Percentage by Difficulty')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return percentages


def calculate_empty_res_pred_percentage(test_results):
    empty_res_pred_count = sum(1 for item in test_results if not item["res_pred"])
    
    total_queries = len(test_results)
    
    percentage_empty_res_pred = (empty_res_pred_count / total_queries) * 100
    
    labels = ['Empty output', 'Non-empty output']
    sizes = [percentage_empty_res_pred, 100 - percentage_empty_res_pred]
    colors = ['lightcoral', 'lightskyblue']
    
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Percentage of Predicted Queries with Empty output')
    plt.show()


def calculate_error_res_pred_percentage(test_results):
    error_res_pred_count = sum(
        1 for item in test_results 
            if isinstance(item["res_pred"], str) and item["res_pred"].startswith("Error executing query:")
    )    
    total_queries = len(test_results)
    
    percentage_error_res_pred = (error_res_pred_count / total_queries) * 100
    
    labels = ['Error output', 'Non-error output']
    sizes = [percentage_error_res_pred, 100 - percentage_error_res_pred]
    colors = ['lightcoral', 'lightskyblue']
    plt.figure(figsize=(5, 5))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Percentage of Predicted Queries with Error output')
    plt.show()


def accuracy_by_db(test_results):
    db_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for result in test_results:
        db_id = result['db_id']
        score = result['score']

        db_stats[db_id]['total'] += 1
        db_stats[db_id]['correct'] += score

    accuracy_by_db = {
        db_id: (stats['correct'] / stats['total']) * 100
        for db_id, stats in db_stats.items()
    }

    db_ids = list(accuracy_by_db.keys())
    accuracies = list(accuracy_by_db.values())

    plt.figure(figsize=(8, 4))
    plt.bar(db_ids, accuracies, color='skyblue')
    plt.xlabel('Data Base')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy Percentage for Each Database')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return accuracies


def calculate_accuracy_by_difficulty_for_specific_db(test_results, db_id):
    difficulty_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for result in test_results:
        if result['db_id'] == db_id:
            difficulty = result['difficulty']
            score = result['score']

            difficulty_stats[difficulty]['total'] += 1
            difficulty_stats[difficulty]['correct'] += score

    accuracy_by_difficulty = {
        difficulty: (stats['correct'] / stats['total']) * 100
        for difficulty, stats in difficulty_stats.items()
    }

    # Plot the results
    difficulties = list(accuracy_by_difficulty.keys())
    accuracies = list(accuracy_by_difficulty.values())

    plt.figure(figsize=(8, 4))
    plt.bar(difficulties, accuracies, color='skyblue')
    plt.xlabel('Difficulty')
    plt.ylabel('Accuracy Percentage')
    plt.title(f'Accuracy Percentage by Difficulty for {db_id}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return accuracies


def accuracy_by_keyword(test_results):
    keywords = [
        "WHERE", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "FULL JOIN",
        "GROUP BY", "ORDER BY", "HAVING", "DISTINCT", "LIMIT", "OFFSET",
        "UNION", "EXISTS", "BETWEEN", "LIKE", "IN", "COUNT", "MAX", "MIN",
        "SUM", "AVG"
    ]
    keyword_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    for test_result in test_results:
        sql_true = test_result['SQL_true']
        score = test_result['score']
        for keyword in keywords:
            keyword_stats[keyword]['total'] += 1
            if keyword in sql_true:
                keyword_stats[keyword]['correct'] += score

    accuracy_by_keyword = {
        keyword: (stats['correct'] / stats['total']) * 100
        for keyword, stats in keyword_stats.items()
    }

    # Filter keywords with accuracy greater than 0
    filtered_accuracy = {k: v for k, v in accuracy_by_keyword.items() if v > 0}

    # Graficar los resultados
    keywords_list = list(filtered_accuracy.keys())
    accuracies = list(filtered_accuracy.values())

    plt.figure(figsize=(8, 4))
    plt.bar(keywords_list, accuracies, color='skyblue')
    plt.xlabel('SQL Keywords')
    plt.ylabel('Accuracy Percentage')
    plt.title('Accuracy Percentage for Each SQL Keyword')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return filtered_accuracy