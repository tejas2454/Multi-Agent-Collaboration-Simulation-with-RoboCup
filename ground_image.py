
from mplsoccer.pitch import Pitch
pitch = Pitch(pitch_color='grass', line_color='white', stripe=True, view ='half')
fig, ax = pitch.draw()
ax.grid(True)
fig.savefig('./soccer_ground.png',pad_inches=0,bbox_inches='tight')
