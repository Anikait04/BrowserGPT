def plan_steps_update(current_plan_step,entire_plan):

    if current_plan_step < len(entire_plan):
        plan_step = entire_plan[current_plan_step]
    else:
        plan_step = "Finish the task"

    return plan_step