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
//
//
// 
// John Allison 20th October 1996

#include "G4Colour.hh"

#include "G4Threading.hh"

#include <algorithm>

namespace {
  auto clamp = [](double rgba){return std::clamp(rgba, 0., 1.);};
}

// clang-format off
G4Colour::G4Colour (G4double r, G4double gr, G4double b, G4double a):
red(clamp(r)), green(clamp(gr)), blue(clamp(b)), alpha(clamp(a))
{}

G4Colour::G4Colour (G4ThreeVector v):
red(clamp(v.x())), green(clamp(v.y())), blue(clamp(v.z())), alpha (1.)
{}

void G4Colour::SetRed   (G4double r)  {red   = clamp(r);}
void G4Colour::SetGreen (G4double gr) {green = clamp(gr);}
void G4Colour::SetBlue  (G4double b)  {blue  = clamp(b);}
void G4Colour::SetAlpha (G4double a)  {alpha = clamp(a);}
// clang-format on

G4Colour::operator G4ThreeVector() {
  return G4ThreeVector(red,green,blue);
}

std::ostream& operator << (std::ostream& os, const G4Colour& c) {
  os << '(' << c.red << ',' << c.green << ',' << c.blue
            << ',' << c.alpha << ')';
  const std::map<G4String, G4Colour>& colourMap = G4Colour::GetMap();
  // Reverse iterator to pick up English spelling of grey!!  :)
  std::map<G4String, G4Colour>::const_reverse_iterator ri;
  for (ri = colourMap.rbegin(); ri != colourMap.rend(); ++ri) {
    if (c == ri->second) {
      os << " (" << ri->first << ')';
      break;
    }
  }
  
  return os;
}

G4bool G4Colour::operator != (const G4Colour& c) const {
  return (red   != c.red)   ||
      (green != c.green) ||
      (blue  != c.blue)  ||
      (alpha != c.alpha);
}

std::map<G4String, G4Colour> G4Colour::fColourMap;
G4bool G4Colour::fInitColourMap = false;

void G4Colour::AddToMap(const G4String& key, const G4Colour& colour)
{
  // Allow only master thread to populate the map
  if (!G4Threading::IsMasterThread()) {
    static G4bool first = true;
    if (first) {
      first = false;
      G4Exception
      ("G4Colour::AddToMap(const G4String& key, const G4Colour& colour)",
       "greps0002", JustWarning,
       "Attempt to add to colour map from non-master thread.");
    }
    return;
  }

  // Add standard colours to map
  InitialiseColourMap();  // Initialises if not already initialised

  // Convert to lower case since colour map is case insensitive
  G4String myKey = G4StrUtil::to_lower_copy(key);
  
  if (fColourMap.find(myKey) == fColourMap.end()) fColourMap[myKey] = colour;
  else {
    G4ExceptionDescription ed; 
    ed << "G4Colour with key " << myKey << " already exists." << G4endl;
    G4Exception
      ("G4Colour::AddToMap(const G4String& key, const G4Colour& colour)",
       "greps0001", JustWarning, ed,
       "Colour key exists");
  }
}

void G4Colour::InitialiseColourMap()
{
  if (fInitColourMap) return;

  fInitColourMap = true;

  // clang-format off
  // Standard colours
  AddToMap("white",   G4Colour::White());
  AddToMap("grey",    G4Colour::Grey());
  AddToMap("gray",    G4Colour::Gray());
  AddToMap("black",   G4Colour::Black());
  AddToMap("brown",   G4Colour::Brown());
  AddToMap("red",     G4Colour::Red());
  AddToMap("green",   G4Colour::Green());
  AddToMap("blue",    G4Colour::Blue());
  AddToMap("cyan",    G4Colour::Cyan());
  AddToMap("magenta", G4Colour::Magenta());
  AddToMap("yellow",  G4Colour::Yellow());
  // clang-format off
}

G4bool G4Colour::GetColour(const G4String& key, G4Colour& result)
// Get colour for given key, placing it in result.
// The key is usually the name of the colour.
// The key is case insensitive.
// Returns false if key doesn't exist, leaving result unchanged.
{
  // Add standard colours to map
  InitialiseColourMap();  // Initialises if not already initialised

  G4String myKey = G4StrUtil::to_lower_copy(key);

  // NOLINTNEXTLINE(modernize-use-auto): Explicitly want a const_iterator
  std::map<G4String, G4Colour>::const_iterator iter = fColourMap.find(myKey); 

  // Don't modify "result" if colour was not found in map
  if (iter == fColourMap.cend()) {
    G4ExceptionDescription ed;
    ed << "Colour \"" << key << "\" not found. No action taken.";
    G4Exception("G4Colour::GetColour", "greps0003", JustWarning, ed);
    return false;
  }

  result = iter->second;

  return true;
}

const std::map<G4String, G4Colour>& G4Colour::GetMap()
{
  // Add standard colours to map
  InitialiseColourMap();  // Initialises if not already initialised
 
  return fColourMap;
}

G4bool G4Colour::operator< (const G4Colour& rhs) const
{
  if (red < rhs.red) return true;
  if (red == rhs.red) {
    if (green < rhs.green) return true;
    if (green == rhs.green) {
      if (blue < rhs.blue) return true;
      if (blue == rhs.blue) {
	      if (alpha < rhs.alpha) return true;
      }
    }
  }
  return false;
}
