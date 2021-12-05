def test(X, y):
    return X,y

def test2(X):
    return X
    
def run_grid(X,Y,grid):
    grid_result = grid.fit(X, y)