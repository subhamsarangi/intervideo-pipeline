from src.database.db import list_questions_by_run

qs = list_questions_by_run(9, 'candidate')
if qs:
    print(f"Type: {type(qs[0])}")
    if hasattr(qs[0], 'keys'):
        print(f"Keys: {list(qs[0].keys())}")
    print(f"Sample: {qs[0]}")
else:
    print("No questions found")
