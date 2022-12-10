from collections import Counter, OrderedDict

from utils import get_index


def remove_records(
    df, X_proba, attributes, sensitive_attributes, sensitive_dict, df_combinations
):
    record_informations = []
    # for every record removed, we append a "record information"
    # a "record information" is a tuple of length == len(sensitive_attributes)
    # each value in the tuple is the value the respective attribute had in the removed record

    # additionally, we'll make sure to remove the record in every list of records we have

    for att in sensitive_attributes:
        #### Removing DN
        for val in sensitive_dict[att]["D"]["values_list"]:
            if (
                len(sensitive_dict[att]["D"][val]["N"])
                > sensitive_dict[att]["D"][val]["N_exp"]
            ):
                to_remove = (
                    len(sensitive_dict[att]["D"][val]["N"])
                    - sensitive_dict[att]["D"][val]["N_exp"]
                )
                for record in sensitive_dict[att]["D"][val]["N"][:to_remove]:
                    record_info = []
                    for att2 in sensitive_attributes:
                        index = get_index(df, att2, attributes)
                        value = record[index]
                        record_info.append(value)
                        if att2 != att:
                            for val2 in sensitive_dict[att2]["D"]["values_list"]:
                                if record in sensitive_dict[att2]["D"][val2]["P"]:
                                    sensitive_dict[att2]["D"][val2]["P"].remove(record)
                                if record in sensitive_dict[att2]["D"][val2]["N"]:
                                    sensitive_dict[att2]["D"][val2]["N"].remove(record)
                            for val2 in sensitive_dict[att2]["P"]["values_list"]:
                                if record in sensitive_dict[att2]["P"][val2]["P"]:
                                    sensitive_dict[att2]["P"][val2]["P"].remove(record)
                                if record in sensitive_dict[att2]["P"][val2]["N"]:
                                    sensitive_dict[att2]["P"][val2]["N"].remove(record)
                    record_informations.append(tuple(record_info))
                    X_proba.remove(record)
                sensitive_dict[att]["D"][val]["N"] = sensitive_dict[att]["D"][val]["N"][
                    to_remove:
                ]

        ### Removing PP
        for val in sensitive_dict[att]["P"]["values_list"]:
            if (
                len(sensitive_dict[att]["P"][val]["P"])
                > sensitive_dict[att]["P"][val]["P_exp"]
            ):
                to_remove = (
                    len(sensitive_dict[att]["P"][val]["P"])
                    - sensitive_dict[att]["P"][val]["P_exp"]
                )
                for record in sensitive_dict[att]["P"][val]["P"][:to_remove]:
                    record_info = []
                    for att2 in sensitive_attributes:
                        index = get_index(df, att2, attributes)
                        value = record[index]
                        record_info.append(value)
                        if att2 != att:
                            for val2 in sensitive_dict[att2]["D"]["values_list"]:
                                if record in sensitive_dict[att2]["D"][val2]["P"]:
                                    sensitive_dict[att2]["D"][val2]["P"].remove(record)
                                if record in sensitive_dict[att2]["D"][val2]["N"]:
                                    sensitive_dict[att2]["D"][val2]["N"].remove(record)
                            for val2 in sensitive_dict[att2]["P"]["values_list"]:
                                if record in sensitive_dict[att2]["P"][val2]["P"]:
                                    sensitive_dict[att2]["P"][val2]["P"].remove(record)
                                if record in sensitive_dict[att2]["P"][val2]["N"]:
                                    sensitive_dict[att2]["P"][val2]["N"].remove(record)
                    record_informations.append(tuple(record_info))
                    X_proba.remove(record)
                sensitive_dict[att]["P"][val]["P"] = sensitive_dict[att]["P"][val]["P"][
                    to_remove:
                ]

    if len(record_informations) == 0:
        print("No records removed! The dataset is already balanced")
        return X_proba, [], sensitive_dict

    print("Records removed:", len(record_informations))
    print("")
    print("Current length of dataset: ", len(X_proba))

    common_combs = Counter(tuple(record_informations))
    common_combs = list(OrderedDict(common_combs.most_common()))

    # the unique set of "record information", ordered by their frequency
    # i.e. every unique combinations of sensitive attributes removed

    for comb in df_combinations:
        if comb not in common_combs:
            common_combs.append(comb)

    # additional unique combinations of sensitive attributes
    # not among those removed
    # we append them to the very end of the list => lower priority
    return X_proba, common_combs, sensitive_dict
