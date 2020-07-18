import numpy as np
import streamlit as st
from linear_algebra import Transform2D as tf2D
from apps import utilities as utils

def main():
    #vectors = np.array([
    #                      [1.0,0.0],
    #                      [0.0,1.0], 
    #                   ])
    vectors = utils.load('apps/vectors2D.pkl')
    
    st.markdown("# Enter the transformation matrix")

    
    ########################      Specify Transformation matrix      ###############################
    label = 'Syntax(witout quotes): "A, B; C, D" or "[A, B; C, D]" or "[[A, B]; [C, D]]" '
    m = st.text_input(label, value="[2.0,  -0.5;   1.0,  0.5]")
    try:
        matrix, status = utils.str2matrix(m)
    except:
        status = False
    if(status==False):
        st.error('[Enter Transformation matrix] Invalid input, input should be like(without quotes): "A, B; C, D" or "[A, B; C, D]" or "[[A, B]; [C, D]]" ')
    ################################################################################################

    ########################      Add Equation       ###############################################
    
    eq_checkbox = st.sidebar.checkbox("Want to add a equation", value = True)
    if eq_checkbox:
        equation = st.sidebar.text_input('Enter equation f(x)=', value="sqrt(9-x^2)")
        count = st.sidebar.number_input("Specify number of data points", value=30)
        range_checkbox = st.sidebar.checkbox("Manually Specify Range", value = True)
        range_ = (-3,3)
        if range_checkbox:
            r = st.sidebar.text_input('Syntax(witout quotes): "A, B" or "[A, B]"', value="[-3, 3]")
            try:
                range_, status = utils.str2vec(r)
            except:
                status = False
            if(status==False):
                st.error('[Add Equation] Invalid input, input should be like(without quotes): "(1, 2)" or "1, 2"')
    info_checkbox = st.sidebar.checkbox("(Click to see)Supported functions", value = False)
    if info_checkbox:
        st.sidebar.info("""
                        Here it support most of the function, like:    
                        sin(x), cos(x), e^(x), log(x), ...    
                        (If a function is supported by numpy you can use it here as well) \n
                        Examples:   
                        f(x) = sin(x) * cos(x)     
                        f(x) = e^(log(x)) + sin(2\*pi\*x)   
                        For a complete list vist **[HERE](https://numpy.org/doc/stable/reference/routines.math.html)**
                        """)
    ################################################################################################

    
    ########################      Add Sliders for Transformation matrix      #######################
    st.sidebar.markdown("-----")
    a,b,c,d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    sliders_tf = st.sidebar.checkbox("Enable Sliders", value = False)
    if sliders_tf:
        st.sidebar.markdown("## Interact with Transformation matrix")
        a = st.sidebar.slider('A', a-10, a+10, a, 0.1)
        b = st.sidebar.slider('B', b-10, b+10, b, 0.1)
        c = st.sidebar.slider('C', c-10, c+10, c, 0.1)
        d = st.sidebar.slider('D', d-10, d+10, d, 0.1)
    ################################################################################################
    
    
    ########################      Display Transformation matrix      ###############################
    st.latex(r'''\begin{bmatrix}
     A & B\\  
     C & D\\
    \end{bmatrix}=
    \begin{bmatrix}
     ''' + str(a) + ''' & ''' + str(b) + '''\\\\
     ''' + str(c) + ''' & ''' + str(d) + '''
    \end{bmatrix}
    
    ''')
    ################################################################################################
    
    
    ########################      Convert Vectors to a displayable string      #####################
    str_vectors = []
    for i in range(len(vectors)):
        str_vectors.append(utils.vec2str(vectors[i]))
    ################################################################################################
    
    st.sidebar.markdown("-----")
    st.sidebar.markdown("## Add/Remove Vectors")
    ########################      Add a Vectors      ###############################################
    label = "[1.0, 2.0]"
    v = st.sidebar.text_input('Add vector,  Syntax(witout quotes): "A, B" or "[A, B]"', value=label)
    if st.sidebar.button("Add"):
        try:
            vector, status = utils.str2vec(v)
        except:
            status = False
        if(status==False):
            st.error('[Add a Vectors] Invalid input, input should be like(without quotes): "[1, 2]" or "1, 2"')
        else:
            vectors = np.vstack((vectors, vector))
            str_vectors.append(utils.vec2str(vector))
    ################################################################################################
    
    ########################      Remove a Vector      ############################################
    remove_vec = st.sidebar.number_input("Remove a vector(Specify index[1 base])      specify -1 to remove last added vector)", value=1)
    remove_vec -= 1
    if st.sidebar.button("Remove"):
        status = utils.str2vec(v)
        if(len(vectors)==0):
            st.error("[Remove a Vectors] There isn't any vector to remove")
        elif(remove_vec == -2):
            vectors = np.delete(vectors, -1, 0)
        elif(remove_vec < 0 or remove_vec >= len(vectors)):
            st.error("[Remove a Vectors] Invalid input, it must be between: 1 and "+str(len(vectors)))
        else:
            vectors = np.delete(vectors, remove_vec, 0)
    
    ################################################################################################
    
    
    ################################      Vectors      #############################################
    st.sidebar.markdown("-----")
    st.sidebar.markdown("## Vectors")
    st.sidebar.markdown("**(you can also change there values)**")
    text_vectors = []
    for i in range(len(vectors)):
        text_vectors.append(st.sidebar.text_input('v'+str(i+1), value=str_vectors[i]))
    
    for i in range(len(text_vectors)):
        vector, status = utils.str2vec(text_vectors[i])
        if(status==False): 
            st.error('[Editing Vector v' + str(i+1) + '] Invalid input, input should be like(without quotes): "[1, 2]" or "1, 2"')
            break
        vectors[i] = vector
    
    ################################################################################################
    ################################################################################################
    ################################################################################################
    
    if (len(vectors)>=1):
        vector_label = st.checkbox("Wanna keep all vectors' labels", value = True)
        transform_matrix = np.array([
                                        [a,b],
                                        [c,d],
                                    ])
        transform = tf2D(transform_matrix, vector_label)
        transform.add_vectors(vectors)
        utils.k_decimal_matrix(transform.transformed_vectors)
        
        if(eq_checkbox):
            eq_status, range_status = transform.add_equation(   
                                                            eq      = equation,
                                                            x_range = range_,
                                                            count   = count, 
                                                            )
            if(eq_status==False):
                st.error("[Add Equation] Error: Equation is not correct")
            elif(range_status==False):
                st.error("[Add Equation] Error: Either equation or range is incorrect.")

        
        orignal_plt, transform_plt, combine_plt = transform.fig()
        st.pyplot(combine_plt)
        st.pyplot(orignal_plt)
        st.pyplot(transform_plt)
    
    
        s = []
        for i in range(len(vectors)):
            temp = r'''    \begin{bmatrix}
         ''' + str(vectors[i,0]) + '''\\\\
         ''' + str(vectors[i,1]) + '''
        \end{bmatrix} \\to \\begin{bmatrix}
         ''' + str(transform.transformed_vectors[i,0]) + '''\\\\
         ''' + str(transform.transformed_vectors[i,1]) + '''
        \\end{bmatrix}'''
            s.append(temp)
        
        st.markdown("## Transformed vectors:")
        for i in range(len(s)):
            st.latex("v_"+str(i+1)+":"+s[i])
    utils.save(vectors, 'apps/vectors2D.pkl')

################################################################################################

if __name__ == "__main__":
    main()