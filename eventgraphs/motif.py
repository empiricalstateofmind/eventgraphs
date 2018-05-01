from collections.abc import Iterable

# Check that undirected motifs are being processed correctly.

class Motif(object):


	def __init__(self,e1,e2, condensed, directed):
		"""
		"""
		self.directed = directed

		for s,nodes in zip(['U1', 'U2', 'V1', 'V2'],
					   [e1[0], e2[0], e1[1], e2[1]]):

			if isinstance(nodes, Iterable) and not isinstance(nodes, str):
				setattr(self, s, set(nodes))
			else:
			 	setattr(self, s, {nodes})

		motif = (len(self.U1 & self.U2),
				 len(self.V1 & self.U2),
				 len(self.U2 - (self.U1 | self.V1)),
				 len(self.U1 & self.V2),
				 len(self.V1 & self.V2),
				 len(self.V2 - (self.U1 | self.V1)))
		
		if condensed:
			motif = tuple(int(bool(entry)) for entry in motif)

		if len(e1)==4:
			self._motif = (*motif, e1[-1], e2[-1])
		else:
			self._motif = motif

		# Cleanup - keep the object lightweight
		for attr in ['U1', 'U2', 'V1', 'V2']:
			delattr(self, attr)
			
	def __str__(self):
		return self._iconify_motif()

	def __hash__(self):
		return hash(self._motif)

	def __repr__(self):
		return "< Motif {} {} >".format(self._motif, self._iconify_motif())

	def __eq__(self, other):
		# Add check to see if other is a motif.

		if isinstance(other, self.__class__):
			if self._motif == other._motif:
				return True
			else:
				return False

		elif isinstance(other, str):
			if self.__str__ == other:
				return True
			else:
				return False

	def _iconify_motif(self):
		"""

		
		Input:

		Returns:
			None
		"""
		icons=['●','○','+'] 
		string = ''
		for ix, entry in enumerate(self._motif[:3]):
			string += icons[ix]*entry

		if self.directed: 
			string +='|'
			for ix, entry in enumerate(self._motif[3:6]):
				string += icons[ix]*entry
			
		if len(self._motif) == 8:
			# We can shorten this or alter it.
			string += " ({},{})".format(self._motif[-2][0]+self._motif[-2][-1],
										self._motif[-1][0]+self._motif[-1][-1])

		return string