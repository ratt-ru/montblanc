from transitions import Machine

from montblanc.impl.biro.common.fsm import FsmModel, states, transitions, TRANSFER_DATA

def get_fsm(composite_solver):
    model = FsmModel(composite_solver)
    return Machine(model=model,
        states=states,
        transitions=transitions,
        initial=TRANSFER_DATA)
