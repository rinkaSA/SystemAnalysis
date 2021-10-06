import numpy as np
import matplotlib.pyplot as plt


def create_A(a1, a2):
    a = np.zeros((3, 3))
    a[0][1] = 1
    a[1][2] = 1
    a[2][0] = -1
    a[2][1] = -a1
    a[2][2] = -a2
    return a


def create_b():
    b = np.zeros((3, 1))
    b[0][0] = 0
    b[1][0] = 0
    b[2][0] = 1
    return b


def create_c():
    c = []
    for i in range(3):
        value = np.zeros((1, 3))
        value[0][i] = 1
        c.append(value)
    return c


def calculate_f(a, t, q):
    f = np.eye(3)
    try:
        if isinstance(q, int) == 0:
            raise Exception('Not integer accuracy!')
        elif q < 2 or q > 10:
            raise Exception("Not allowed accuracy!")
    except Exception as e:
        print(e)
        return 0
    for i in range(1, q + 1):
        f += np.linalg.matrix_power(a.dot(t), i) / np.math.factorial(i)
    return f


def calculate_g(a, q, t0):
    b = create_b()
    e = np.eye(3)
    tmp = e * t0
    for j in range(2, q + 1):
        tmp += (np.linalg.matrix_power(a, j - 1).dot(e) * (t0 ** j)) / np.math.factorial(j)

    return tmp.dot(b)


def calculate_equation_1(f, x, u, g):
    res = f.dot(x) + g * u
    return res


def calculate_equation_2(x, c, y):
    y.append(c.dot(x))
    return y


def get_vector_x():
    x = np.zeros((3, 1))
    x1 = input('Input x1: ')
    x[0][0] = x1
    x2 = input('Input x2: ')
    x[1][0] = x2
    x3 = input('Input x3: ')
    x[2][0] = x3
    return x


def variables():
    a1 = float(input('Input a1 for matrix A: '))
    a2 = float(input('Input a2 for matrix A: '))
    t0 = float(input('Input period kvantuvannya t0: '))
    q = int(input('Input accuracy q: '))
    var = [a1, a2, t0, q]
    return var


def interface():
    val = variables()
    x = get_vector_x()
    t = list(np.arange(0, 50 + val[2], val[2]))
    a = create_A(val[0], val[1])
    f = calculate_f(a, val[2], val[3])
    g = calculate_g(a, val[3], val[2])
    seq_of_x = []
    seq_of_x.append(x)
    opt = int(input('Input option 1/2/3: '))
    if opt == 1:
        u = float(input("Input u:"))
        for i in range(len(t)):
            x = calculate_equation_1(f, x, u, g)
            seq_of_x.append(x)
    elif opt == 2:
        k0 = float(input("Input k0: "))
        k = 0
        u = 1
        for i in range(len(t)):
            x = calculate_equation_1(f, x, u, g)
            seq_of_x.append(x)
            print(u)
            if k == k0:
                u = -1
            k += 1
    elif opt == 3:
        k0 = float(input("Input k0: "))
        u = 1
        k = 0
        for i in range(len(t)):
            x = calculate_equation_1(f, x, u, g)
            seq_of_x.append(x)
            if k == k0:
                u = -1
            elif k == 2 * k0:
                u = -1
            elif k == 3 * k0:
                u = 1
            k += 1

    else:
        print("There is no such type of task")
        return 0
    c = create_c()
    y1 = []
    y2 = []
    y3 = []
    for i in seq_of_x:
        y1 = calculate_equation_2(i, c[0], y1)
        y2 = calculate_equation_2(i, c[1], y2)
        y3 = calculate_equation_2(i, c[2], y3)

    lol1 = []
    for i in y1:
        a = i.tolist()
        lol1.append(a[0])
    list_y1 = []
    for m in range(len(lol1) - 1):
        list_y1.append(lol1[m][0])

    lol2 = []
    for w in y2:
        a = w.tolist()
        lol2.append(a[0])
    list_y2 = []
    for s in range(len(lol2) - 1):
        list_y2.append(lol2[s][0])

    lol3 = []
    for n in y3:
        s = n.tolist()
        lol3.append(s[0])
    y3_listed = []
    for p in range(len(lol3) - 1):
        y3_listed.append(lol3[p][0])

    plt.xlabel('t')
    plt.ylabel('x1(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + 1, 5))
    plt.plot(t, list_y1, color='purple')
    plt.show()
    plt.xlabel('t')
    plt.ylabel('x2(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + 1, 5))
    plt.plot(t, list_y2, color='purple')
    plt.show()
    plt.xlabel('t')
    plt.ylabel('x3(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + 1, 5))
    plt.plot(t, y3_listed, color='purple')
    plt.show()


interface()
