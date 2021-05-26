from cx_Freeze import setup, Executable
import os
os.environ['TCL_LIBRARY'] = r'C:\\ProgramData\\Anaconda3\\envs\\ch2_venv\\tcl\\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\\ProgramData\\Anaconda3\\envs\\ch2_venv\\tcl\\tk8.6'
 
base = None   
 
executables = [Executable("SQLAlc.py", base=base)]
 
packages = ["idna","sqlalchemy", "urllib","pyodbc"]
options = {
    'build_exe': {    
        'packages':packages,
    },    
}
 
setup(
    name = "my_name",
    options = options,
    version = "0.1",
    description = 'My discription',
    executables = executables
)