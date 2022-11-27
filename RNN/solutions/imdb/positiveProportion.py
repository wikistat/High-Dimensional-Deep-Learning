plt.figure(figsize = (8,3))

plt.subplot(1,3,1)
sns.countplot(x=y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("y_train")

plt.subplot(1,3,3)
sns.countplot(x=y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("y_test")


unique,  counts = np.unique(y_train, return_counts = True)
print("y_train distribution: ", dict(zip(unique,counts)))

unique,  counts = np.unique(y_test, return_counts = True)
print("y_test distribution: ", dict(zip(unique,counts)))