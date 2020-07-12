import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import Transform as tf


def input_vector(s: str):
    s = s.replace(" ", "")
    s = s.replace("[", "")
    s = s.replace("]", "")
    for i in range(len(s)):
        if(s[i].isnumeric() or s[i]==',' or s[i]=='-'): 
            continue
        else:
            return None, False
    l = s.split(",")
    return np.array([int(l[0]), int(l[1])]), True
    
def str_vec(v):
    s = "[" + str(v[0]) + ", " + str(v[1]) + "]"
    return s

def save(vectors):
    with open('vectors.pkl','wb') as f:
        pickle.dump(vectors, f)
def load():
    with open('vectors.pkl','rb') as f:
        vectors  = pickle.load(f)
    return vectors

def main():
    #vectors = np.array([
    #                      [1,1],
    #                      [-1,2],
    #                   ])
    vectors = load()
    
    st.write("Enter the transformation matrix")
    st.latex(r'''\begin{bmatrix}
     A & B\\  
     C & D\\
    \end{bmatrix}''')
    
    #st.latex(r'''
    #a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
    #\sum_{k=0}^{n-1} ar^k =
    #a \left(\frac{1-r^{n}}{1-r}\right)
    #''')

    a = st.number_input("A", value=1)
    b = st.number_input("B", value=0)
    c = st.number_input("C", value=0)
    d = st.number_input("D", value=1)
    
    st.sidebar.latex(r'''\begin{bmatrix}
     ''' + str(a) + ''' & ''' + str(b) + '''\\\\
     ''' + str(c) + ''' & ''' + str(d) + '''
    \end{bmatrix}''')
    
    a = st.sidebar.slider('A', a-20, a+20, a, 1)
    b = st.sidebar.slider('B', b-20, b+20, b, 1)
    c = st.sidebar.slider('C', c-20, c+20, c, 1)
    d = st.sidebar.slider('D', d-20, d+20, d, 1)
    
    transform_matrix = np.array([
                                    [a,b],
                                    [c,d],
                                ])
    
    
    str_vectors = []
    for i in range(len(vectors)):
        str_vectors.append(str_vec(vectors[i]))
    
    #"""             Add a Vector             """
    if(len(vectors)!=0):
        label = str_vec(vectors[-1])
    else:
        label = "[1, 2]"
    v = st.text_input("Add more vectors", value=label)
    if st.button("Add more vectors"):
        try:
            vector, status = input_vector(v)
        except:
            status = False
        if(status==False):
            st.error('Invalid input, input should be like(without quotes): "[1, 2]" or "1, 2"')
        else:
            vectors = np.vstack((vectors, vector))
            save(vectors)
            str_vectors.append(str_vec(vector))
    
    #"""             Remove a Vector             """
    remove_vec = st.number_input("Remove a vector(Specify index[1 base])      specify -1 to remove last added vector)", value=1)
    remove_vec -= 1
    if st.button("Remove"):
        status = input_vector(v)
        if(len(vectors)==0):
            st.error("There isn't any vector to remove")
        elif(remove_vec == -2):
            vectors = np.delete(vectors, -1, 0)
            save(vectors)
        elif(remove_vec < 0 or remove_vec >= len(vectors)):
            st.error("Invalid input, it must be between: 1 and "+str(len(vectors)))
        else:
            vectors = np.delete(vectors, remove_vec, 0)
            save(vectors)
        
    text_vectors = []
    for i in range(len(vectors)):
        text_vectors.append(st.sidebar.text_input('v'+str(i+1), value=str_vectors[i]))
    
    for i in range(len(text_vectors)):
        vector, status = input_vector(text_vectors[i])
        if(status==False): 
            st.error('Invalid input, input should be like(without quotes): "[1, 2]" or "1, 2"')
            break
        vectors[i] = vector
    
    if (len(vectors)>=1):
        transform = tf.Transform(transform_matrix)
        transform.add_vectors(vectors)
        orignal_plt, transform_plt = transform.fig()
        st.pyplot(orignal_plt)
        st.pyplot(transform_plt)
    
    


if __name__ == "__main__":
    main()