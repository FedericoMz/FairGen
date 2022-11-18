from record_removal import remove_records, get_index
from constraints import get_constraints
from genetic import GA

from sklearn_extra.cluster import KMedoids
import lightgbm as ltb
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from sklearn.naive_bayes import GaussianNB

import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from itertools import product


class FairGen:
    def __init__(
        self,
        df,
        sensitive_attributes,
        class_name,
        causal_reg=[],
        causal_class=[],
        discrete_attributes=[],
        values_in_dataset_attributes=[],
        mode="distance",
        ds="Fixed",
    ):

        self.df = df
        self.sensitive_attributes = sensitive_attributes
        self.class_name = class_name
        self.causal_reg = causal_reg
        self.causal_class = causal_class
        self.discrete_attributes = discrete_attributes
        self.values_in_dataset_attributes = values_in_dataset_attributes
        self.mode = mode
        self.ds = ds

        self.genetic_data = []

        self.values_in_dataset_indexes = []

        self.discrete_indexes = []

        self.regular_indexes = []

        self.causal_reg_attributes = []

        self.causal_class_attributes = []

        # global attributes
        self.attributes = [col for col in self.df.columns if col != self.class_name]

        self.val_comb = []
        for att in self.sensitive_attributes:
            self.val_comb.append(list(self.df[att].unique()))

        self.df_combinations = list(product(*self.val_comb))

        print("Ranking the data...")

        self.X_proba = self.__ranker()

        edges = []

        if len(self.causal_reg) > 0:

            for e in causal_reg:

                X_index = []
                for feat in e[0]:
                    X_index.append(get_index(self.df, feat, self.attributes))
                    edges.append((feat, e[1]))
                y_index = get_index(self.df, e[1], self.attributes)
                self.causal_reg_attributes.append(e[1])

                regressor = self.__get_causal_regressor(self.X_proba, X_index, y_index)

                e[0] = X_index
                e[1] = y_index
                e.append(regressor)

                # Each e in causual has:
                # e[0] = name of ind variable
                # e[1] = name of dep variable
                # we turned e[0] and e[1] into the variable indexes and we added
                # e[2] = predictor

        if len(causal_class) > 0:

            for e in causal_class:

                X_index = []
                for feat in e[0]:
                    X_index.append(get_index(self.df, feat, self.attributes))
                    edges.append((feat, e[1]))
                y_index = get_index(self.df, e[1], self.attributes)
                self.causal_class_attributes.append(e[1])

                classifier = self.__get_causal_classifier(
                    self.X_proba, X_index, y_index
                )

                e[0] = X_index
                e[1] = y_index
                e.append(classifier)

        nodes = self.df.columns

        dag = nx.DiGraph(edges)

        dag.add_nodes_from(nodes)

        nx.draw_networkx(
            dag,
            pos=nx.circular_layout(dag),
            font_size=10,
            node_size=350,
            node_color="#abdbe3",
        )

        plt.title("Assumed Ground Truth", fontsize=13)

        plt.show()

        self.values = self.X_proba.copy()

        self.target = None

        print("Creating the sensitive dictionary...")

        self.sensitive_dict = self.__get_discrimination(
            self.X_proba, sensitive_attributes, class_name
        )

        self.X_proba = self.X_proba.iloc[
            :, :-1
        ]  # removing the last column (with the ranker proba)

        self.og_df = self.X_proba.copy()  ###ADDED

        lof = LOF()
        forest = IForest()

        lof.fit(self.X_proba)
        forest.fit(self.X_proba)

        self.X_proba = self.X_proba.values

        for att in discrete_attributes:
            self.discrete_indexes.append(get_index(self.df, att, self.attributes))

        for att in values_in_dataset_attributes:
            self.values_in_dataset_indexes.append(
                get_index(self.df, att, self.attributes)
            )

        for att in self.attributes:
            if (
                att
                not in sensitive_attributes
                + self.causal_reg_attributes
                + self.causal_class_attributes
            ):
                self.regular_indexes.append((get_index(self.df, att, self.attributes)))

    def execute(self):

        self.X_proba, self.common_combs, self.sensitive_dict = remove_records(
            self.df,
            self.X_proba,
            self.attributes,
            self.sensitive_attributes,
            self.sensitive_dict,
            self.df_combinations,
        )

        constraints = get_constraints(
            self.df,
            self.attributes,
            self.sensitive_attributes,
            self.sensitive_dict,
            self.common_combs,
        )

        for const in constraints.keys():
            subset = self.og_df.copy()
            for tup in const:
                subset = subset[subset[subset.columns[tup[0]]] == tup[1]]

            if len(subset) > 0:
                kmedoids = KMedoids(n_clusters=1, random_state=42).fit(subset)
                medoid = kmedoids.cluster_centers_[0]
            else:
                medoid = [None]
            #### /ADDED

            new_records = GA(
                self.values,
                const,
                constraints[const],
                self.forest,
                medoid,
                self.values_in_dataset_indexes,
                self.discrete_indexes,
                self.regular_indexes,
                self.causal_reg,
                self.causal_class,
                self.mode,
                self.ds,
            )

            for all_records in new_records:
                for record in all_records:
                    target = const[0][1]
                    for tup in const[1:]:
                        att = self.attributes[tup[0]]
                        val = tup[1]
                        if val in self.sensitive_dict[att]["D"]["values_list"]:
                            if target == 0:
                                self.sensitive_dict[att]["D"][val]["N"].append(record)
                            else:
                                self.sensitive_dict[att]["D"][val]["P"].append(record)
                        elif val in self.sensitive_dict[att]["P"]["values_list"]:
                            if target == 0:
                                self.sensitive_dict[att]["P"][val]["N"].append(record)
                            else:
                                self.sensitive_dict[att]["P"][val]["P"].append(record)
                        else:
                            print("ERROR! ERROR!")
                            print("Value", val, "shouldn't exist for attribute", att)

                    self.X_proba.append(record)
                    self.genetic_data.append(record)
        print("=== NEW DATASET ===")
        final_df = pd.DataFrame.from_records(self.X_proba)
        final_df.columns = self.attributes + [self.class_name]

        genetic_df = pd.DataFrame.from_records(self.genetic_data)
        genetic_df.columns = self.ttributes + [self.class_name]

        self.get_discrimination(final_df, self.sensitive_attributes, self.class_name)

        print("")
        print("OG dataset length:", len(self.df))
        print("Records generated:", len(genetic_df))
        print("New dataset length:", len(final_df))
        return final_df, genetic_df

    def __ranker(self):

        X = self.df[self.attributes].values
        y = self.df[self.class_name].to_list()
        clf = GaussianNB()
        # clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100)
        clf.fit(X, y)
        X_proba = [None] * len(X)

        for k in range(len(X)):
            X_proba[k] = np.append(X[k], (y[k], clf.predict_proba(X)[k][1]))

        X_proba = np.array(X_proba, dtype=float).tolist()

        X_proba_list = []

        for arr in self.X_proba:
            X_proba_list.append(arr[: len(self.attributes) + 2])

        X_proba = pd.DataFrame.from_records(X_proba_list)
        X_proba.columns = self.attributes + [self.class_name] + ["proba"]

        return X_proba

    def __get_discrimination(self):

        # ASSUMPTION: Each value of the attribute is discriminated
        # For each value, we therefore apply the Preferential Sampling formulas to compute the discrimination
        # If discrimination > 0, the assumption holds true
        # Otherwhise, it doesn't. This means the value is actually *privileged*
        # A dictionary of sensitive attributes and values is created as such
        #
        # Please note the sum of records to add / remove for each priviliged value
        # should be equal to the sum of records to add / removed for each discriminated value
        #
        # A rounding error is possible

        sensitive_dict = {}

        tot_disc = 0
        tot_pos = 0

        # df = X_proba

        for attr in self.sensitive_attributes:
            print()
            print("Analizing", attr, "...")
            sensitive_dict[attr] = {}
            sensitive_dict[attr]["D"] = {}
            sensitive_dict[attr]["P"] = {}
            sensitive_dict[attr]["D"]["values_list"] = []
            sensitive_dict[attr]["P"]["values_list"] = []
            values = self.df[attr].unique()
            for val in values:
                PP = self.df[
                    (self.df[attr] != val) & (self.df[self.class_name] == 1)
                ].values.tolist()
                PN = self.df[
                    (self.df[attr] != val) & (self.df[self.class_name] == 0)
                ].values.tolist()
                DP = self.df[
                    (self.df[attr] == val) & (self.df[self.class_name] == 1)
                ].values.tolist()
                DN = self.df[
                    (self.df[attr] == val) & (self.df[self.class_name] == 0)
                ].values.tolist()

                disc = len(DN) + len(DP)
                priv = len(PN) + len(PP)
                pos = len(PP) + len(DP)
                neg = len(PN) + len(DN)

                DP_exp = round(disc * pos / len(self.df))
                PP_exp = round(priv * pos / len(self.df))
                DN_exp = round(disc * neg / len(self.df))
                PN_exp = round(priv * neg / len(self.df))

                discrimination = len(PP) / (len(PP) + len(PN)) - len(DP) / (
                    len(DP) + len(DN)
                )

                if discrimination >= 0:
                    status = "D"
                    sensitive_dict[attr][status][val] = {}
                    print("")
                    print(val, "is discriminated:", discrimination)

                    sensitive_dict[attr][status][val]["P"] = sorted(
                        DP, key=lambda x: x[len(DP[0]) - 1]
                    )
                    sensitive_dict[attr][status][val]["P_exp"] = DP_exp
                    sensitive_dict[attr][status][val]["P_curr"] = 0

                    for i in range(len(sensitive_dict[attr][status][val]["P"])):
                        del sensitive_dict[attr][status][val]["P"][i][-1]

                    sensitive_dict[attr][status][val]["N"] = sorted(
                        DN, key=lambda x: x[len(DN[0]) - 1], reverse=True
                    )
                    sensitive_dict[attr][status][val]["N_exp"] = DN_exp
                    sensitive_dict[attr][status][val]["N_curr"] = 0

                    for i in range(len(sensitive_dict[attr][status][val]["N"])):
                        del sensitive_dict[attr][status][val]["N"][i][-1]

                    print(
                        "- DP:",
                        len(sensitive_dict[attr][status][val]["P"]),
                        "· Expected:",
                        DP_exp,
                        "· To be added:",
                        abs(len(DP) - DP_exp),
                    )
                    print(
                        "- DN:",
                        len(sensitive_dict[attr][status][val]["N"]),
                        "· Expected:",
                        DN_exp,
                        "· To be removed:",
                        abs(len(DN) - DN_exp),
                    )

                    tot_disc = tot_disc + abs(len(DP) - DP_exp)

                else:
                    status = "P"
                    sensitive_dict[attr][status][val] = {}
                    print("")
                    print(val, "is privileged:", discrimination)

                    sensitive_dict[attr][status][val]["P"] = sorted(
                        DP, key=lambda x: x[len(DP[0]) - 1]
                    )
                    sensitive_dict[attr][status][val]["P_exp"] = DP_exp
                    sensitive_dict[attr][status][val]["P_curr"] = 0

                    for i in range(len(sensitive_dict[attr][status][val]["P"])):
                        del sensitive_dict[attr][status][val]["P"][i][-1]

                    sensitive_dict[attr][status][val]["N"] = sorted(
                        DN, key=lambda x: x[len(DN[0]) - 1], reverse=True
                    )
                    sensitive_dict[attr][status][val]["N_exp"] = DN_exp
                    sensitive_dict[attr][status][val]["N_curr"] = 0

                    for i in range(len(sensitive_dict[attr][status][val]["N"])):
                        del sensitive_dict[attr][status][val]["N"][i][-1]

                    print(
                        "- PP:",
                        len(sensitive_dict[attr][status][val]["P"]),
                        "· Expected:",
                        DP_exp,
                        "· To be removed:",
                        abs(len(DP) - DP_exp),
                    )
                    print(
                        "- PN:",
                        len(sensitive_dict[attr][status][val]["N"]),
                        "· Expected:",
                        DN_exp,
                        "· To be added:",
                        abs(len(DN) - DN_exp),
                    )

                    tot_pos = tot_pos + abs(len(DP) - DP_exp)

                sensitive_dict[attr][status]["values_list"].append(val)

        round_error = abs(tot_disc - tot_pos)

        if round_error > 0:
            print("")
            print("Due to a rounding error, the final dataset will be slightly smaller")
        return sensitive_dict

    def __get_causal_classifier(self, data, index_X, index_y):

        X = data.iloc[:, index_X].values
        y = data.iloc[:, [index_y]].values
        classifier = ltb.LGBMClassifier().fit(X, y)

        return classifier

    def __get_causal_regressor(self, data, index_X, index_y):
        X = data.iloc[:, index_X].values
        y = data.iloc[:, [index_y]].values
        regr = ltb.LGBMRegressor().fit(X, y)

        return regr

