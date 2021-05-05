import numpy as np
# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d') 
    ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            \n
            '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

if __name__ == '__main__':
    # Define name for output file
    output_file = 'dump.ply'
    pcd = np.load("dump.npy").astype(np.float32)
    one = np.ones((pcd.shape[0],3))
    one = np.float32(one)*255
    # Generate point cloud
    print("\n Creating the output file... \n")
#    create_output(points_3D, colors, output_file)
    create_output(pcd, one, output_file)
