import pandas as pd

def get_true_positives(df, prediction_column, validation_column):
    positives = df[df[prediction_column] < 0]
    true_positives = positives[positives[validation_column]  == 1]
    return true_positives

def get_false_positives(df, prediction_column, validation_column):
    positives = df[df[prediction_column] < 0]
    false_positives = positives[positives[validation_column] == -1]
    return false_positives

def get_true_negatives(df, prediction_column, validation_column):
    positives = df[df[prediction_column] > 0]
    true_negatives = positives[positives[validation_column] == -1]
    return true_negatives

def get_false_negatives(df, prediction_column, validation_column):
    positives = df[df[prediction_column] > 0]
    false_negatives = positives[positives[validation_column] == 1]
    return false_negatives

def get_f1_score(df, prediction_column, validation_column):
    TP = len(get_true_positives(df, prediction_column, validation_column))
    FP  = len(get_false_positives(df, prediction_column, validation_column))
    FN = len(get_false_negatives(df, prediction_column, validation_column))
    precision = float(TP)/float(TP + FP)
    recall = float(TP)/float(TP + FN)
    return (2 * precision * recall) / (precision + recall)