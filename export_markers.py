import bpy
import os
D = bpy.data

save_dir = 'C:\\Users\\windis\\Documents\\MSc_Habib\\gradient_ansatz\\data\\tracking_data'
nb_frames = 200

for clip in D.movieclips:
    for track in clip.tracking.tracks:
        fn = os.path.join(save_dir, '{0}_{1}.csv'.format(clip.name.split('.')[0], track.name))
        with open(fn, 'w') as f:
            for frameno in range(nb_frames):
                markerAtFrame = track.markers.find_frame(frameno)
                if markerAtFrame:
                    coords = markerAtFrame.co.xy
                    f.write('{0} {1} {2}\n'.format(frameno, coords[0], coords[1]))
                