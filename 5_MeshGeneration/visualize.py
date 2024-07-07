import vtk

def visualize_mesh_from_obj(obj_file):
    # Read the obj file
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update()
    
    # Get the mesh data
    mesh = reader.GetOutput()
    
    # Create mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)
    
    # Create actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Set background to dark gray
    
    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("Mesh Visualization")
    
    # Create interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Initialize and start the interactor
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

# Example usage
obj_file_path = r'C:\Users\csyas\OneDrive\Desktop\projects\3D_SFM\5_MeshGeneration\denseMesh.obj'

# Visualize the mesh
visualize_mesh_from_obj(obj_file_path)
