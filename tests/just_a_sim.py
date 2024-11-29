# Simulating injection 

def vulnerable_function():
    # Attackers inputs something malicious
    user_input = input("Enter a malicious command: ")
    exec(user_input)  # Will execute without validation or checks 
  
def test_injection():
    # Simulate their input
    malicious_input = "__import__('os').system('touch harmless_file')"

    try:
        # test injection of attackers input 
        exec(malicious_input)
        print("Vulnerability tested: Command executed successfully.")
    except Exception as e:
        # error handling 
        print(f"Error caught: {e}")
