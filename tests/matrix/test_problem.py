from compute_physics import print_header
from compute_physics.matrix.algorithm import partial_pivoting_gauss, lu, solve_by_lu
import numpy as np

def verify_solution(A,x,b,tol=1e-10) -> bool:
    print("verify solution:")
    res = np.dot(A,x) - b
    res_norm = np.linalg.norm(res)

    print(f"residual vector: {res}") 
    print(f"residual norm: {res_norm}") 
    
    if res_norm < tol:
        print(f"Pass with tolerance {tol}\n")
        return True
    else:
        print(f"Fail with tolerance {tol}\n")
        return False

def verify_lu(A,P,L,U,tol=1e-10) -> bool:
    print("verify lu decomposition:")
    PA = np.dot(P,A)
    LU = np.dot(L,U)
    res_norm = np.linalg.norm(PA-LU)

    print(f"residual norm: {res_norm}") 
    
    if res_norm < tol:
        print(f"Pass with tolerance {tol}\n")
        return True
    else:
        print(f"Fail with tolerance {tol}\n")
        return False
    

def test_problem2_2():
    print_header("2.2")
    A = np.array([[2,3,5],[3,4,8],[1,3,3]], dtype=float)
    b = np.array([5,6,5],dtype=float)
    x = partial_pivoting_gauss(A, b)
    print(f"answer\n{x}")
    assert verify_solution(A,x,b)
    
def test_problem_2_3():
    print_header("2.3")
    A = np.array([[4,2,-2],[2,2,-2],[-2,-3,13]], dtype=float)
    P,L,U = lu(A)
    print(f"answer:\nP:\n{P}\nL:\n{L}\nU:\n{U}")
    assert verify_lu(A,P,L,U)

def test_problem_2_4():
    print_header("2.4")
    A = np.array([[4,2,-2],[2,2,-2],[-2,-3,13]], dtype=float)
    b = np.array([8,4,5],dtype=float)
    P,L,U =  lu(A)
    x = solve_by_lu(P,L,U,b)
    print(f"answer:\n{x}")
    assert verify_solution(A,x,b)

def test_problem_2_5():
    print_header("2.5")
    A = np.array([[2,2,3],[4,7,7],[-2,4,5]], dtype=float)
    b = np.array([3,1,-7],dtype=float)
    P,L,U = lu(A)
    print(f"answer:\nP:\n{P}\nL:\n{L}\nU:\n{U}")
    assert verify_lu(A,P,L,U)
    x = solve_by_lu(P,L,U,b)
    print(f"answer:\n{x}")
    assert verify_solution(A,x,b)