import streamlit as st
from apps.linear_transform import tf2D, tf3D 

def main():
    st.sidebar.info('''
                    GitHub:   **[yuvraj97](https://github.com/yuvraj97)**    
                    Linkedin: **[yuvraj97](https://www.linkedin.com/in/yuvraj97/)**    
                    Contribute *[HERE](https://github.com/yuvraj97/Linear-Algebra-Visualization)*
                    ''')
    option = st.selectbox("3D/2D", ("3D", "2D"))
    if(option=="2D"):
        tf2D.main()
    else:
        tf3D.main()


if __name__ == "__main__":
    main()