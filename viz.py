from sklearn import metrics

def confusion_matrix_heat_map(clf):
    """This function takes a classifier clf as an argument to create a confusion matrix.  The function then normalizes the data across the confusion matrix and corrects a known bug in matplotlib that incorrectly cuts off the top and bottom rows of the heat map."""
    # create confusion matrix <cm>
    cm = metrics.confusion_matrix(clf.predict(X_test), y_test)
    # create normalized confusion matrix <cm_nor>
    cm_nor = np.zeros((cm.shape[0], cm.shape[1]))
    for col in range(cm.shape[1]):
        cm_nor[:,col] = (cm[:,col] / sum(cm[:,col]))
    plt.ylim(-10, 10)
    # create normalized confusion matrix heat map
    sns.heatmap(cm_nor, cmap="Blues", annot=True,annot_kws={"size": 8})
    locs, labels = plt.xticks()
    plt.xticks(locs, ("DEM", "REP", "UNA"))
    locs, labels = plt.yticks()
    plt.yticks(locs, ("DEM", "REP", "UNA"))
    plt.yticks(rotation = 0)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual Voter Party Change")
    # known bug in matplotlib chops off a portion of the
    # top and bottom rows of heat maps.  This section of
    # code recovers the top and bottom limits and moves them
    # so that the map displays appropriately.  
    b, t = plt.ylim() 
    b += 0.5 
    t -= 0.5 
    plt.ylim(b, t) 
    plt.show() 
