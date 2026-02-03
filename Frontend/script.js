let tracks = [];

function loadPlaylist() {
  fetch('http://localhost:5000/playlist')
    .then(res => res.json())
    .then(data => {
      tracks = data.items.map(item => item.track);
      displayTracks(tracks);
    });
}

function displayTracks(trackList) {
  const trackContainer = document.getElementById('tracks');
  trackContainer.innerHTML = '';
  trackList.forEach(track => {
    const div = document.createElement('div');
    div.className = 'track';
    div.innerText = `${track.name} by ${track.artists.map(a => a.name).join(', ')}`;
    trackContainer.appendChild(div);
  });
}

function filterTracks() {
  const query = document.getElementById('search').value.toLowerCase();
  const filtered = tracks.filter(track =>
    track.name.toLowerCase().includes(query) ||
    track.artists.some(artist => artist.name.toLowerCase().includes(query))
  );
  displayTracks(filtered);
}