from criteria import MSESplitFinder


class TreeNode:
    def __init__(self, cond, feature, lvl):
        self.condition = cond
        self.condition_feature = feature
        self.level = lvl
        self.prediction = None
        self.left = None
        self.right = None

    def next(self, x_i):
        return (
            self.left if (x_i[self.condition_feature] < self.condition) else self.right
        )


class Tree:
    def __init__(self, label, max_depth=2, n_features=None,
                 split_finder=None, leaf_fn=None):
        self.label = label
        self.max_depth = max_depth
        self.n_features = n_features
        self.split_finder = split_finder or MSESplitFinder()
        self.leaf_fn = leaf_fn or (lambda y: y.mean())
        self.root = None

    def find_split(self, df):
        return self.split_finder.find_split(df, self.label, self.n_features)

    def build_tree(self, df, curr):
        if curr is None or curr.condition is None or curr.condition_feature is None:
            return
        curr.prediction = self.leaf_fn(df[self.label])
        if curr.level == self.max_depth:
            return
        left_subset = df.loc[df[curr.condition_feature] < curr.condition]
        left_cond, _, left_feature = self.find_split(left_subset)
        if left_cond is not None and left_feature is not None:
            curr.left = TreeNode(left_cond, left_feature, curr.level + 1)
            self.build_tree(left_subset, curr.left)

        right_subset = df.loc[df[curr.condition_feature] >= curr.condition]
        right_cond, _, right_feature = self.find_split(right_subset)
        if right_cond is not None and right_feature is not None:
            curr.right = TreeNode(right_cond, right_feature, curr.level + 1)
            self.build_tree(right_subset, curr.right)

    def fit(self, df):
        cond, _, feature = self.find_split(df)
        if cond is not None and feature is not None:
            self.root = TreeNode(cond, feature, 0)
            self.build_tree(df, self.root)

    def predict_one(self, x_i):
        node = self.root
        while node and (node.left or node.right):
            next_node = node.next(x_i)
            if next_node is None:
                break
            node = next_node
        return None if node is None else node.prediction

    def predict(self, df):
        return df.apply(lambda row: self.predict_one(row), axis=1)

    def print_conditions(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node is None:
            return
        if node.condition is None or node.condition_feature is None:
            print(f"{indent}<no split> (level={node.level}, pred={node.prediction})")
            return
        print(
            f"{indent}{node.condition_feature} < {node.condition:.4f} "
            f"(level={node.level}, pred={node.prediction})"
        )
        if node.left or node.right:
            self.print_conditions(node.left, indent + "  ")
            self.print_conditions(node.right, indent + "  ")
