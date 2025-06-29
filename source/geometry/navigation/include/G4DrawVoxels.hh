//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
// 
// G4DrawVoxels
//
// Class description:
//
// Utility class for the visualization of voxels in the detector geometry.

// Original author: L.G., 29 July 1999
// --------------------------------------------------------------------
#ifndef G4DrawVoxels_HH
#define G4DrawVoxels_HH 1

#include "G4VisAttributes.hh"
#include "G4VoxelLimits.hh"
#include "G4PlacedPolyhedron.hh"

class G4SmartVoxelHeader;
class G4LogicalVolume;

/**
 * @brief G4DrawVoxels is a utility class for the visualization of voxels
 * in the detector geometry.
 */

class G4DrawVoxels
{
  public:

    /**
     * Constructor. It initialises the members data to default colors.
     */
    G4DrawVoxels();

    /**
     * Copy constructor and assignment operator not allowed.
     */
    G4DrawVoxels(const G4DrawVoxels&) = delete;	
    G4DrawVoxels operator=(const G4DrawVoxels&) = delete;	

    /**
     * Default Destructor.
     */
    ~G4DrawVoxels() = default;
    
    /**
     * Draws voxels for the specified logical volume.
     */
    void DrawVoxels(const G4LogicalVolume* lv) const;

    /**
     * Creates polyhedra for the specified logical volume.
     */
    G4PlacedPolyhedronList* CreatePlacedPolyhedra(const G4LogicalVolume*) const;

    /**
     * Visualisation attributes control. Allow changing colors of the drawing.
     */
    void SetVoxelsVisAttributes(G4VisAttributes&,
                                G4VisAttributes&,
                                G4VisAttributes&);
    void SetBoundingBoxVisAttributes(G4VisAttributes&);

  private:

    void ComputeVoxelPolyhedra(const G4LogicalVolume*,
                               const G4SmartVoxelHeader*,
                                     G4VoxelLimits&,
                                     G4PlacedPolyhedronList*) const;
    
  private:
  
    G4VisAttributes fVoxelsVisAttributes[3];
    G4VisAttributes fBoundingBoxVisAttributes;
};

#endif
