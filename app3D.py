# SessionState module from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import numpy as np
import streamlit as st
import pickle
import Transform as tf


def input_vector(s: str):
    remove = set((' ', '[' , ']' , '(' , ')' , '\n'))
    allowed = set((',', '-', '.'))
    new_s = []
    for c in s:
        if c not in remove:
            new_s.append(c)
        
    new_s = "".join(new_s)
    for c in new_s:
        if(c.isnumeric() or c in allowed): 
            continue
        else:
            return None, False
    
    l = new_s.split(",")
    for i in range(len(l)):
        l[i] = float(l[i])
    return np.array(l), True
    
def input_tf_matrix(s: str):
    remove = set((' ', '[' , ']' , '(' , ')' , '\n'))
    allowed = set((',', '-', ';', '.'))
    new_s = []
    for c in s:
        if c not in remove:
            new_s.append(c)
    new_s = "".join(new_s)
    for c in new_s:
        if(c.isnumeric() or c in allowed): 
            continue
        else:
            return None, False
    
    l = new_s.split(";")
    for i in range(3):
        l[i] = l[i].split(",")
        for j in range(3):
            l[i][j] = float(l[i][j])
    return np.array(l), True

def reduce_to_k_decimal(n,k=3):
    return int(n * 10**k)/10**k

def mat_dec(m,k=3):
    for i in range(len(m)):
        for j in range(3):
            m[i][j] = reduce_to_k_decimal(m[i][j],k)
        
def str_vec(v):
    s = "[" + str(v[0]) + ", " + str(v[1]) + ", " + str(v[2]) +"]"
    return s

def save(vectors):
    with open('vectors3D.pkl','wb') as f:
        pickle.dump(vectors, f)
        
def load():
    with open('vectors3D.pkl','rb') as f:
        vectors  = pickle.load(f)
    return vectors

def main():
    #vectors = np.array([
    #                      [1.0, 0.0, 0.0],
    #                      [0.0, 1.0, 0.0], 
    #                      [0.0, 0.0, 1.0], 
    #                   ])
    vectors = load()
    
    st.markdown("# Enter the transformation matrix")

    
    ########################      Specify Transformation matrix      ###############################
    label = '''Syntax(witout quotes): "[a11, a12, a13 ;   a21, a22, a23 ;   a31, a32, a33]"'''
    m = st.text_area(label, value='''[   [-1.0,  0.0,  0.0]; 
     [0.0,  -1.0,  0.0]; 
     [0.0,   0.0,  -1.0]   ]''')
    try:
        matrix, status = input_tf_matrix(m)
    except:
        status = False
    if(status==False):
        st.error(''' **Error** *[Enter Transformation matrix]* Invalid input, input should be like:    
            $[$    
            $\ \ \ \ \ \ [a_{11}, a_{12}, a_{13}]\\color{black}{\\textbf{;}}$     
            $\ \ \ \ \ \ [a_{21}, a_{22}, a_{23}]\\color{black}{\\textbf{;}}$     
            $\ \ \ \ \ \ [a_{31}, a_{32}, a_{33}]$     
            $]$''')
    ################################################################################################

    ########################      Add Equation       ###############################################
    
    eq_checkbox = st.sidebar.checkbox("Want to add a equation", value = True)
    if eq_checkbox:
        equation = st.sidebar.text_input('Enter equation f(x, y)=', value="sqrt[9 - (x-0)^2 - (y-0)^2] + 0")
        opacity = st.sidebar.slider('opacity', 0.2, 1.0, 0.6, 0.05)
        count = st.sidebar.number_input("Specify number of data points", value=30)
        if(count>100):
            st.sidebar.warning('''**Warning** *[Specify number of data points]*,    
                       More then 100 data points are not allowed, so count is set to 100''')
            count =100
        range_checkbox = st.sidebar.checkbox("Manually Specify Range", value = True)
        if range_checkbox:
            r = st.sidebar.text_input('Syntax(witout quotes): "min, max" or "[min, max]"', value="[-3, 3]")
            try:
                range_, status = input_vector(r)
            except:
                status = False
            if(status==False):
                st.sidebar.error(''' **Error** *[Add Equation]*      
                         Invalid input, input should be like(without quotes): "(1, 2)" or "1, 2"''')
            elif(len(range_)!=2): 
                st.sidebar.error(''' **Error** *[Add a Vectors]*    
                         You specified more or less then 2 numbers to specify range.     
                         Expected input is(without quotes): "[1, 2]" or "1, 2"''')
        else:
            range_ = ()
            
    info_checkbox = st.sidebar.checkbox("(View) Supported functions", value = True)
    if info_checkbox:
        st.sidebar.info("""
                        Here it support most of the function, like:    
                        sin(x), cos(x), e^(x), log(x), ...    
                        (If a function is supported by numpy you can use it here as well) \n
                        Examples:   
                        f(x, y) = sin(x) * cos(y)    
                        f(x, y) = e^(log(x)) + sin(2\*pi\*y)   
                        For a complete list vist **[HERE](https://numpy.org/doc/stable/reference/routines.math.html)**
                        """)
    ################################################################################################

    
    ########################      Add Sliders for Transformation matrix      #######################
    st.sidebar.markdown("-----")
    sliders_tf = st.sidebar.checkbox("Enable Sliders", value = False)
    sliders_matrix = np.empty(shape=matrix.shape)
    if sliders_tf:
        st.sidebar.markdown("## Interact with Transformation matrix")
        for i in range(3):
            for j in range(3):
                sliders_matrix[i][j] = st.sidebar.slider('a'+str(i+1)+str(j+1), matrix[i][j]-10, matrix[i][j]+10, matrix[i][j], 0.5)
        matrix = sliders_matrix
    ################################################################################################
    
    
    ########################      Display Transformation matrix      ###############################
    st.latex(r'''\begin{bmatrix}
     a_{11} & a_{12} & a_{13}\\  
     a_{21} & a_{21} & a_{21}\\
     a_{31} & a_{31} & a_{31}\\
    \end{bmatrix}=
    \begin{bmatrix}
     ''' + str(matrix[0][0]) + ''' & ''' + str(matrix[0][1]) + ''' & ''' + str(matrix[0][2]) + '''\\\\
     ''' + str(matrix[1][0]) + ''' & ''' + str(matrix[1][1]) + ''' & ''' + str(matrix[1][2]) + '''\\\\
     ''' + str(matrix[1][0]) + ''' & ''' + str(matrix[2][1]) + ''' & ''' + str(matrix[2][2]) + '''
    \end{bmatrix}
    
    ''')
    ################################################################################################
    
    
    ########################      Convert Vectors to a displayable string      #####################

    ################################################################################################
    
    st.sidebar.markdown("-----")
    st.sidebar.markdown("## Add/Remove Vectors")
    ########################      Add a Vectors      ###############################################
    label = "[1.0, 2.0, 3.0]"
    v = st.sidebar.text_input('Add vector,  Syntax(witout quotes): "a, b, c" or "[a, b, c]"', value=label)
    if st.sidebar.button("Add"):
        try:
            vector, status = input_vector(v)
        except:
            status = False
        if(status==False):
            st.sidebar.error('''  **Error** *[Add a Vectors]*     
                     Invalid input, input should be like(without quotes): "[1, 2, 3]" or "1, 2, 3"''')
        elif(len(vector)!=3):
            st.sidebar.error(''' **Error** *[Add a Vectors]*     
                     You specified *more* or *less* then **3** numbers to specify vector.     
                     Expected input is(without quotes): "**[1.0, 2.0, 3.0]**" or "**1.0, 2.0, 3.0**"''')
        else:
            vectors = np.vstack((vectors, vector))
            #save(vectors)
    ################################################################################################
    
    ########################      Remove a Vector      ############################################
    remove_vec = st.sidebar.number_input("Remove a vector(Specify index[1 base])      specify -1 to remove last added vector)", value=1)
    remove_vec -= 1
    if st.sidebar.button("Remove"):
        status = input_vector(v)
        if(len(vectors)==0):
            st.sidebar.error('''  **Error** *[Remove a Vectors]*     
                     There isn't any vector to remove''')
        elif(remove_vec == -2):
            vectors = np.delete(vectors, -1, 0)
            #save(vectors)
        elif(remove_vec < 0 or remove_vec >= len(vectors)):
            st.sidebar.error(''' **Error** *[Remove a Vectors]*       
                     Invalid input, it must be between: 1 and '''+str(len(vectors)))
        else:
            print("BEFORE:", vectors)
            vectors = np.delete(vectors, remove_vec, 0)
            print("After:", vectors)
            #save(vectors)
    
    ################################################################################################
    
    
    ################################      Vectors      #############################################
    st.sidebar.markdown("-----")
    st.sidebar.markdown("## Vectors")
    st.sidebar.markdown("**(you can also change there values)**")
    print(vectors)
    for i in range(len(vectors)):
        text_vector = st.sidebar.text_input('v'+str(i+1), value=str_vec(vectors[i]))
        vector, status = input_vector(text_vector)
        if(status==False): 
            st.sidebar.error(''' **Error** *[Editing Vector v''' + str(i+1) + ''']*      
                     Invalid input, input should be like(without quotes): "**[1.0, 2.0, 3.0]**" or "**1.0, 2.0, 3.0**"''')
            break
        elif(len(vector)!=3):
            st.sidebar.error('''  **Error** *[Add a Vectors]*    
                     You specified *more* or *less* then **3** numbers to specify vector.     
                     Expected input is(without quotes): "**[1.0, 2.0, 3.0]**" or "**1.0, 2.0, 3.0**"''')
        else:
            vectors[i] = vector
    
    ################################################################################################
    ################################################################################################
    ################################################################################################
    
    if (len(vectors)>=1):
        #surface_label = st.checkbox("Want to keep all vectors' labels", value = True)
        transform = tf.Transform3D(matrix)
        transform.add_vectors(vectors)
        mat_dec(transform.transformed_vectors)
        if(eq_checkbox):
            eq_status, range_status = transform.add_equation(   
                                                            eq      = equation,
                                                            range_  = range_,
                                                            count   = count, 
                                                            opacity = opacity,
                                                            )
            if(eq_status==False):
                st.error(''' **Error** *[Add Equation]*     
                         Equation is not correct     
                         **Examples**:   
                         f(x, y) = sin(x) * cos(y)    
                         f(x, y) = e^(log(x)) + sin(2\*pi\*y)   
                         For a complete list vist **[HERE](https://numpy.org/doc/stable/reference/routines.math.html)**
                         ''')
            elif(range_status==False):
                st.error(''' **Error** *[Add Equation]*      
                         Either equation or range is incorrect.''')

        
        orignal_plt, transform_plt, combine_plt = transform.fig()
        st.plotly_chart(combine_plt)
        st.plotly_chart(orignal_plt)
        st.plotly_chart(transform_plt)
        #transform.show()
        #print(vectors)
        #print(transform.transformed_vectors)
        #print()
    
        s = []
        for i in range(len(vectors)):
            temp = r'''    \color{blue}{\begin{bmatrix}
         ''' + str(vectors[i,0]) + '''\\\\
         ''' + str(vectors[i,1]) + '''\\\\
         ''' + str(vectors[i,2]) + '''
        \end{bmatrix}} \\to \color{red}{\\begin{bmatrix}
         ''' + str(transform.transformed_vectors[i,0]) + '''\\\\
         ''' + str(transform.transformed_vectors[i,1]) + '''\\\\
         ''' + str(transform.transformed_vectors[i,2]) + '''
        \\end{bmatrix}}'''
            s.append(temp)
        
        st.markdown("## Transformed vectors:")
        for i in range(len(s)):
            st.latex("v_"+str(i+1)+":"+s[i])
    save(vectors)
################################################################################################

if __name__ == "__main__":
    main()