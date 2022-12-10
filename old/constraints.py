from utils import get_index
from collections import Counter


def get_constraints(df, attributes, sensitive_attributes, sensitive_dict, common_combs):

    constraints = []
    # a constraint is a binary tuple
    # tuple[0] = index of a feature; tuple[1] = value of the feature
    # we'll make a constraint for the sensitive attribute(s) and the target variable

    # "N_cur" => current length of the respective subset
    for att in sensitive_attributes:
        for val in sensitive_dict[att]["D"]["values_list"]:
            sensitive_dict[att]["D"][val]["N_curr"] = len(
                sensitive_dict[att]["D"][val]["N"]
            )
            sensitive_dict[att]["D"][val]["P_curr"] = len(
                sensitive_dict[att]["D"][val]["P"]
            )
            DN_curr = sensitive_dict[att]["D"][val]["N_curr"]
            DN_exp = sensitive_dict[att]["D"][val]["N_exp"]
            DP_curr = sensitive_dict[att]["D"][val]["P_curr"]
            DP_exp = sensitive_dict[att]["D"][val]["P_exp"]
            print(
                att,
                val,
                "DN_cur",
                DN_curr,
                "DN_exp",
                DN_exp,
                "DP_cur",
                DP_curr,
                "DP_exp",
                DP_exp,
            )
        for val in sensitive_dict[att]["P"]["values_list"]:
            sensitive_dict[att]["P"][val]["N_curr"] = len(
                sensitive_dict[att]["P"][val]["N"]
            )
            sensitive_dict[att]["P"][val]["P_curr"] = len(
                sensitive_dict[att]["P"][val]["P"]
            )
            PN_curr = sensitive_dict[att]["P"][val]["N_curr"]
            PN_exp = sensitive_dict[att]["P"][val]["N_exp"]
            PP_curr = sensitive_dict[att]["P"][val]["P_curr"]
            PP_exp = sensitive_dict[att]["P"][val]["P_exp"]
            print(
                att,
                val,
                "PN_cur",
                PN_curr,
                "PN_exp",
                PN_exp,
                "PP_cur",
                PP_curr,
                "PP_exp",
                PP_exp,
            )

    # combinations of sensitive attributes values are ordered according to their frequency in common_combs
    # higher frequency == higher priority
    # for each value in the comb, we check if a new record with *that* value is needed
    # (i.e, if N_curr < N_exp or P_curr < P_exp)
    # if every value in the combination do pass the check (either for a negative record or a positive record)
    # we'll make a constraint with those values (and the target variable)
    # we repeat those steps as long as neither a negative or a positive record is needed
    # then we'll try the next combination
    # this way, before creating records with a less frequent combination of sens attributes values,
    # we are sure we exhausted the more frequent combinations
    #

    for comb in common_combs:
        while True:
            starting_const_len = len(constraints)
            ok_comb_neg = True
            ok_comb_pos = True
            constraint_neg = [(len(attributes), 0)]
            constraint_pos = [(len(attributes), 1)]
            for i in range(len(comb)):
                att = sensitive_attributes[i]
                val = comb[i]

                if val in sensitive_dict[att]["D"]["values_list"]:
                    if (
                        sensitive_dict[att]["D"][val]["N_curr"]
                        < sensitive_dict[att]["D"][val]["N_exp"]
                    ):
                        constraint_neg.append((get_index(df, att, attributes), val))
                    else:
                        ok_comb_neg = False
                elif val in sensitive_dict[att]["P"]["values_list"]:
                    if (
                        sensitive_dict[att]["P"][val]["N_curr"]
                        < sensitive_dict[att]["P"][val]["N_exp"]
                    ):
                        constraint_neg.append((get_index(df, att, attributes), val))
                    else:
                        ok_comb_neg = False

                if val in sensitive_dict[att]["D"]["values_list"]:
                    if (
                        sensitive_dict[att]["D"][val]["P_curr"]
                        < sensitive_dict[att]["D"][val]["P_exp"]
                    ):
                        constraint_pos.append((get_index(df, att, attributes), val))
                    else:
                        ok_comb_pos = False
                elif val in sensitive_dict[att]["P"]["values_list"]:
                    if (
                        sensitive_dict[att]["P"][val]["P_curr"]
                        < sensitive_dict[att]["P"][val]["P_exp"]
                    ):
                        constraint_pos.append((get_index(df, att, attributes), val))
                    else:
                        ok_comb_pos = False

            if ok_comb_neg == True:
                constraints.append(tuple(constraint_neg))
                for tup in constraint_neg[1:]:
                    att = attributes[tup[0]]
                    val = tup[1]
                    if val in sensitive_dict[att]["D"]["values_list"]:
                        sensitive_dict[att]["D"][val]["N_curr"] = (
                            sensitive_dict[att]["D"][val]["N_curr"] + 1
                        )
                    elif val in sensitive_dict[att]["P"]["values_list"]:
                        sensitive_dict[att]["P"][val]["N_curr"] = (
                            sensitive_dict[att]["P"][val]["N_curr"] + 1
                        )

            if ok_comb_pos == True:
                constraints.append(tuple(constraint_pos))
                for tup in constraint_pos[1:]:
                    att = attributes[tup[0]]
                    val = tup[1]
                    if val in sensitive_dict[att]["D"]["values_list"]:
                        sensitive_dict[att]["D"][val]["P_curr"] = (
                            sensitive_dict[att]["D"][val]["P_curr"] + 1
                        )
                    elif val in sensitive_dict[att]["P"]["values_list"]:
                        sensitive_dict[att]["P"][val]["P_curr"] = (
                            sensitive_dict[att]["P"][val]["P_curr"] + 1
                        )

            if ok_comb_neg == False and ok_comb_pos == False:
                break

            if len(constraints) == starting_const_len:
                break

    """print("XXX Printing stuff for debugging purposes...")
    for att in sensitive_attributes:
        for val in sensitive_dict[att]["D"]["values_list"]:
            DN_curr = sensitive_dict[att]["D"][val]["N_curr"]
            DN_exp = sensitive_dict[att]["D"][val]["N_exp"]
            DP_curr = sensitive_dict[att]["D"][val]["P_curr"]
            DP_exp = sensitive_dict[att]["D"][val]["P_exp"]
            print(
                att,
                val,
                "DN_cur",
                DN_curr,
                "DN_exp",
                DN_exp,
                "DP_cur",
                DP_curr,
                "DP_exp",
                DP_exp,
            )
        for val in sensitive_dict[att]["P"]["values_list"]:
            PN_curr = sensitive_dict[att]["P"][val]["N_curr"]
            PN_exp = sensitive_dict[att]["P"][val]["N_exp"]
            PP_curr = sensitive_dict[att]["P"][val]["P_curr"]
            PP_exp = sensitive_dict[att]["P"][val]["P_exp"]
            print(
                att,
                val,
                "PN_cur",
                PN_curr,
                "PN_exp",
                PN_exp,
                "PP_cur",
                PP_curr,
                "PP_exp",
                PP_exp,
            )"""

    constraints = Counter(tuple(constraints))

    return constraints
