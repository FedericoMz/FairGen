def get_index(df, attribute, attributes):
    for i in range(len(attributes)):
        if attributes[i] == attribute:
            return i
