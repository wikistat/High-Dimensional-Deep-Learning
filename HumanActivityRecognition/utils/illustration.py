import matplotlib.pyplot as plt
import numpy as np

## ACP##

def plot_variance_acp(fig, ax, acp, X_acp): 
    ax.bar(range(10), acp.explained_variance_ratio_[:10]*100, align='center',
        color='grey', ecolor='black')
    ax.set_xticks(range(10))
    ax.set_ylabel("Variance")
    ax.set_title("", fontsize=35)
    ax.set_title("Pourcentage de variance expliquee \n des premieres composantes", fontsize=20)

    ax = fig.add_subplot(1,2,2)
    box=ax.boxplot(X_acp[:,0:10])
    ax.set_title("Distribution des premieres composantes", fontsize=20)
    
    
def plot_projection_acp(fig, ax, acp, X_acp, colors=None, labels=None, markersizes=None):
    
    N = X_acp.shape[0]
    if colors==None:
        colors = ["black"]*N
    if labels==None:
        labels = [""]*N
    if markersizes==None:
        markersizes = [1]*N
    
    for x1, x2, c, l, m in zip(X_acp[:,0], X_acp[:,1], colors, labels, markersizes):
        if l=="":
            ax.plot(x1,x2,marker=".", color= c, markersize = m)
        else:
            ax.plot(x1,x2,marker=".", color=c, label=l, markersize = m, linestyle = None)
   
    ax.set_title("Projection des invididus sur les \n  deux premieres composantes", fontsize=20)
    ax.set_xlim(X_acp[:,0].min()-5,X_acp[:,0].max()+5)
    ax.set_ylim(X_acp[:,1].min()-1,X_acp[:,1].max()+1)
    ax.set_xlabel("PC1 : %.2f %%" %(acp.explained_variance_ratio_[0]*100), fontsize=15)
    ax.set_ylabel("PC2 : %.2f %%" %(acp.explained_variance_ratio_[1]*100), fontsize=15)

    plt.legend(fontsize=15, ncol=2)


    ax = fig.add_subplot(1,2,2)
    for f,(x1, x2) in enumerate(zip(acp.components_[0], acp.components_[1])):
        ax.text(x1,x2,str(f), horizontalalignment="center", verticalalignment="center", color="red")
    ax.set_xlim(acp.components_[0].min()-0.002,acp.components_[0].max()+0.002)
    ax.set_ylim(acp.components_[1].min()-0.01,acp.components_[1].max()+0.01)
    ax.set_title("Projection des features sur les \n  deux premisres composantes", fontsize=20)
    
    

#Anomaly Detection#

def plot_decision_function(fig, ax, method, X, y_pred, s=40, method_name="SVM", colors=None, labels=None, markersizes=None):
    N = X.shape[0] 
    if colors==None:
        colors = ["black"]*N
    if labels==None:
        labels = [""]*N
    if markersizes==None:
        markersizes = [1]*N
    
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-5, X[:,0].max()+5, 500), 
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 500))
    
    # plot the line, the points, and the nearest vectors to the plane
    if method_name=="LOF":
        Z = method._decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = method.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    Z_lim = 0 if method_name=="SVM" else method.threshold_
    ax.contourf(xx, yy, Z, levels=[Z.min(), Z_lim, Z.max()], cmap=plt.cm.PuBu)
    
    if method_name!="IF":
        a = ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')


    for x1, x2, c, l, ms, yp in zip(X[:,0], X[:,1], colors, labels, markersizes, y_pred):
        marker = "." if yp==1 else "X"
        if l=="":
            ax.plot(x1,x2,marker=marker, color= c, markersize = ms)
        else:
            ax.plot(x1,x2,marker=marker, color=c, label=l, markersize = ms, linestyle = None)

    plt.axis('tight')
    ax.set_xlim(X[:,0].min()-5,X[:,0].max()+5)
    ax.set_ylim(X[:,1].min()-1,X[:,1].max()+1)
    plt.legend()
    
    
def plot_detection_result(fig, ax, CT, color_dic, s1=70, s2=150, normal_behaviour="WALKING"):
    y_lim = (-1.13,1.13)
    x_lim = [-5,815]
    N = CT.shape[0]
    CT_Normal = CT[CT.Anomaly==normal_behaviour]
    CT_Anormal = CT[CT.Anomaly!=normal_behaviour]
    N_anormal = CT_Anormal.shape[0]
    ax.scatter(CT_Normal.index, -CT_Normal.pred, s = s1, color="white", edgecolors="black", label = "Normal")
    ax.plot(x_lim, [0,0], color="black")

    xticks = []
    xtickslabel = []
    colors = []

    for i,(pred, label) in enumerate(CT_Anormal.values):
        x = int(i/N_anormal*N)
        xticks.append(x)
        xtickslabel.append(label)
        color = color_dic[label]
        colors.append(color)
        ax.plot([x,x] , y_lim, color=color, linestyle="dashed")
        color = "red" if pred==1 else "green"
        ax.scatter(x, -pred, s = s2, color=color, edgecolors="black")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtickslabel, rotation=45, fontsize=15)
    for xtick, color in zip(ax.get_xticklabels(), colors):
        xtick.set_color(color)
    ax.set_xlim(*x_lim)
    ax.set_ylim(y_lim)
