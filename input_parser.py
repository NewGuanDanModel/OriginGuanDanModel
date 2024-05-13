from typing import List, Tuple

PASS_LEGAL = {"PASS", "p", "P", "Pass"}
SHOW_LEGAL = {"SHOW", "Show"}

def same_lists(l1 : List, l2 : List) -> bool:
    if len(l1) != len(l2):
        return False
    length = len(l1)
    if length == 0:
        return True
    
    dummy = [False] * length
    for i in range(length):
        for j in range(length):
            if not dummy[j]:
                if l1[i] == l2[j]:
                    dummy[j] = True
                    break
    
    if dummy.count(False) > 0:
        return False
    return True

def index_of_action(user_choice : List, action_list : List) -> int:
    index = 0
    for index in range(len(action_list)):
        if same_lists(user_choice, action_list[index][2]):
            return index
    return -1

def parse_input(user_input : str, action_list : List) -> Tuple[bool, int]:
    if user_input == None or len(user_input) == 0:
        print("Empty input! Please enter your choice again!")
        return (False, -1)
    if user_input[0] == '[' and user_input[-1] == ']':
        size = len(user_input)
        user_input = user_input[1:size - 1]
        user_choice = user_input.split(",")
        if user_choice[0] in PASS_LEGAL:
            for i in range(len(action_list)):
                if action_list[i][2] == "PASS":
                    return (True, i)
            print("You cannot choose PASS in this turn!")
            return (False, -1)
        elif user_choice[0] in SHOW_LEGAL:
            return (True, -1)
        else:
            index_of_choice = index_of_action(user_choice, action_list)
            if index_of_choice == -1:
                print("There is no such legal choice! Please enter your choice again!")
                return (False, -1)
            else:
                # print(f"Choice is {action_list[index_of_choice]}")
                return (True, index_of_choice)
    else:
        print("Pleae input your choice in this format: [your choice]")
        return (False, -1)