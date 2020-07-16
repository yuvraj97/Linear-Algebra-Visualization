import streamlit as st
import app2D
import app3D

def main():
    option = st.selectbox("3D/2D", ("3D", "2D"))
    if(option=="2D"):
        app2D.main()
    else:
        app3D.main()

if __name__ == "__main__":
    main()